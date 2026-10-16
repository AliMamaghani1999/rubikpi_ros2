import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import threading
import math, os, time

WAYPOINTS_FILE = os.path.expanduser('~/ros2_ws/src/rubikpi_ros2/robot_control/waypoints.txt')

# Motor commands, same values as the W and A/D keys in keyboard_control.py
DRIVE_L, DRIVE_R = 0.3, 0.25    # W: forward
TURN_SPEED = 0.3      
correction_factor=0.75          # A/D: spin in place (L=-0.3,R=0.3 is left)

# CALIBRATE THESE on the floor: open loop accuracy depends entirely on them
LINEAR_SPEED = 0.30     # m/s the robot actually moves with DRIVE_L/DRIVE_R
ANGULAR_SPEED = 2.0*0.7        # rad/s the robot actually spins LEFT with TURN_SPEED
ANGULAR_SPEED_RIGHT = 2.0 *0.86   # rad/s the robot actually spins RIGHT with TURN_SPEED

PAUSE = 0.5             # s to stop between moves so the robot settles

# waypoints (1 = second line of waypoints.txt) the robot reverses into
BACKWARD_WAYPOINTS = {4}
# extra turn time when turning toward a waypoint (1.2 = turn 20% more)
TURN_EXTRA = {4: 1.2}
# waypoints where the robot only makes a quick stop (PAUSE), no turn to the angle
QUICK_STOP_WAYPOINTS = {5}


class WaypointControllerNode(Node):
    def __init__(self):
        super().__init__('waypoint_controller_node')
        
        # create publisher for motor commands
        self.publisher = self.create_publisher(
            Float32MultiArray,
            'motor_commands',
            10
        )
        
        # motor state
        self.current_L = 0.0
        self.current_R = 0.0
        self.running = True

        # setup timer to publish commands at 20Hz
        self.timer = self.create_timer(0.05, self.publish_motor_commands)  # 20Hz

        self.get_logger().info('Waypoint Controller Node started')

        # follow waypoints in a separate thread (timer keeps publishing)
        self.waypoint_thread = threading.Thread(target=self.follow_waypoints, daemon=True)
        self.waypoint_thread.start()

    def load_waypoints(self):
        """read x, y, angle lines from the waypoints file"""
        waypoints = []
        with open(WAYPOINTS_FILE) as f:
            for line in f:
                if line.strip():
                    x, y, theta = (float(v) for v in line.split(','))
                    waypoints.append((x, y, theta))
        return waypoints
    
    def move(self, L, R, duration):
        """send one motor command for duration seconds, then stop and settle"""
        self.current_L = L
        self.current_R = R
        time.sleep(duration)
        self.current_L = 0.0
        self.current_R = 0.0
        time.sleep(PAUSE)
    
    def turn(self, angle, extra=1.0):
        """spin in place by angle radians (positive = left / counter-clockwise)"""
        angle = math.atan2(math.sin(angle), math.cos(angle))  # wrap to [-pi, pi]
        if abs(angle) < 0.01:
            return
        if angle > 0:
            self.move(-TURN_SPEED, TURN_SPEED, abs(angle) / ANGULAR_SPEED * extra)
        else:
            self.move(TURN_SPEED, -TURN_SPEED, abs(angle) / ANGULAR_SPEED_RIGHT * extra)
    
    def drive(self, distance, backward=False):
        """drive straight forward (or backward) distance meters"""
        if distance < 0.01:
            return
        if backward:
            self.move(-DRIVE_L, -DRIVE_R, distance / LINEAR_SPEED*correction_factor)
        else:
            self.move(DRIVE_L, DRIVE_R, distance / LINEAR_SPEED*correction_factor)
    
    def follow_waypoints(self):
        """open loop: turn toward each waypoint, drive to it, turn to its angle"""
        waypoints = self.load_waypoints()
        self.get_logger().info(f'Loaded {len(waypoints)} waypoints from {WAYPOINTS_FILE}')
        time.sleep(1.0)  # give motor_control time to connect
        
        # the robot starts at the first waypoint
        x, y, theta = waypoints[0]
        for i, (wx, wy, wtheta) in enumerate(waypoints[1:], start=1):
            if not self.running:
                return
            self.get_logger().info(f'Waypoint {i}: x={wx}, y={wy}, angle={wtheta}')
            distance = math.hypot(wx - x, wy - y)
            if distance >= 0.01:
                heading = math.atan2(wy - y, wx - x)
                backward = i in BACKWARD_WAYPOINTS
                if backward:
                    heading += math.pi  # face away from the waypoint, then reverse
                self.turn(heading - theta, TURN_EXTRA.get(i, 1.0))
                theta = heading
                self.drive(distance, backward)
            if i in QUICK_STOP_WAYPOINTS:
                wtheta = theta  # keep heading, no turn
            self.turn(wtheta - theta)
            x, y, theta = wx, wy, wtheta
        
        self.get_logger().info('All waypoints reached, shutting down...')
        self.running = False
        rclpy.shutdown()
    
    def publish_motor_commands(self):
        """Publish motor commands to ROS topic"""
        if self.running and rclpy.ok():
            try:
                msg = Float32MultiArray()
                msg.data = [self.current_L, self.current_R]
                self.publisher.publish(msg)
                
                # Log only when commands change to reduce console spam
                if hasattr(self, '_last_L') and hasattr(self, '_last_R'):
                    if self._last_L != self.current_L or self._last_R != self.current_R:
                        self.get_logger().info(f'Motor command: L={self.current_L:.2f}, R={self.current_R:.2f}')
                else:
                    self.get_logger().info(f'Motor command: L={self.current_L:.2f}, R={self.current_R:.2f}')
                
                self._last_L = self.current_L
                self._last_R = self.current_R
            except Exception as e:
                # Context may be shutting down, ignore errors
                if self.running:
                    self.get_logger().error(f'Error publishing command: {str(e)}')
    
    def destroy_node(self):
        """Cleanup operations when node is destroyed"""
        self.running = False
        
        # Send stop command before shutdown
        msg = Float32MultiArray()
        msg.data = [0.0, 0.0]
        self.publisher.publish(msg)
        self.get_logger().info('Stop command sent')
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WaypointControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # If timer already destroyed the node, skip
        if not getattr(node, "_already_destroyed", False):
            try:
                node.destroy_node()
            except Exception:
                pass

        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass
        return 0  # <-- ensure exit code 0



if __name__ == '__main__':
    main()
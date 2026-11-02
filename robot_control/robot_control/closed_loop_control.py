import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from tf2_msgs.msg import TFMessage
import numpy as np
import threading
import csv, math, os, time

WAYPOINTS_FILE = os.path.expanduser('~/ros2_ws/src/rubikpi_ros2/robot_control/hw2_waypoints.txt')
LOG_DIR = os.path.expanduser('~/HW2/logs')

# Tag map: frame name -> world x (m), y (m), direction the tag faces (rad)
TAGS = {
    'tag_0': (1.2, 0.0, math.pi),
    'tag_1': (1.0, 2.2, -math.pi / 2),
    'tag_2': (0.5, 2.0, 0.0),
    'tag_3': (-0.2 * math.cos(math.atan2(2, 1)), -0.2 * math.sin(math.atan2(2, 1)), math.atan2(2, 1)),
}
CAMERA_FORWARD = 0.09   # m the camera sits ahead of the robot center

# Motor commands, same values as the W and A/D keys in keyboard_control.py
DRIVE_L, DRIVE_R = 0.3, 0.25    # W: forward
TURN_SPEED = 0.3                # A/D: spin in place (L=-0.3,R=0.3 is left)

# CALIBRATE THESE on the floor: they carry the estimate when no tag is visible
LINEAR_SPEED = 0.40         # m/s the robot actually moves with DRIVE_L/DRIVE_R
ANGULAR_SPEED_LEFT = 1.4    # rad/s the robot actually spins LEFT with TURN_SPEED
ANGULAR_SPEED_RIGHT = 1.72  # rad/s the robot actually spins RIGHT with TURN_SPEED

# Controller
POS_TOL = 0.03                  # m: close enough to a waypoint
ANGLE_TOL = math.radians(3)     # rad: close enough to a heading
MAX_HEADING_ERROR = math.radians(25)    # stop driving and turn if further off
HEADING_GAIN = 0.4              # steering command per rad of heading error
LINE_GAIN = 2.0                 # how hard to aim back at the line, per m off it
MAX_STEER = 0.10                # largest steering command
SLOW_DIST = 0.50                # m from the waypoint where the robot slows down
MIN_SPEED_FACTOR = 0.5          # slowest fraction of DRIVE_L/DRIVE_R
MIN_PULSE = 0.06                # s: shortest spin the motors respond to
MAX_TURN_TRIES = 8              # spin + look attempts per turn
REFINE_TOL = 0.05               # m: if a stop is further off than this, creep closer
REFINE_TRIES = 2                # creep attempts per waypoint
CREEP_TIMEOUT = 3.0             # s: longest single creep
PAUSE = 0.7                     # s to stop between moves so tags can be read
SPIN_SETTLE = 0.3               # s after a spin before tag readings are trusted
SPIN_UNCERTAINTY = 0.25         # a timed spin can be off by this fraction of its angle
FRESH_TAG_WAIT = 1.5            # s to wait for a usable tag reading after a spin

# When no tag is visible
TAG_TIMEOUT = 2.0               # s without a tag before the robot stops to look
SEARCH_PULSE = 0.15             # s of each search spin
SEARCH_TRIES = 40               # search spins before giving up

# How unsure the motor-command estimate gets while driving, after 1 m
DRIVE_POS_STD = 0.05            # m
DRIVE_ANGLE_STD = 0.12          # rad

# Measurement noise of a tag reading (range, bearing, tag direction)
RANGE_STD = 0.01
RANGE_STD_PER_M = 0.015
BEARING_STD = 0.02
TAG_ANGLE_STD = 0.05
TAG_ANGLE_STD_PER_M = 0.10
TAG_DELAY = 0.25    # s a tag reading is old by the time it arrives
GATE = 16.3     # chi-square, 3 degrees of freedom, p = 0.001
MAX_REJECTS = 10    # consecutive rejected readings before trusting the estimate less
REJECT_POS_STD = 0.15                   # m of uncertainty added after that
REJECT_ANGLE_STD = math.radians(10)     # rad of uncertainty added after that


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class ClosedLoopControllerNode(Node):
    def __init__(self):
        super().__init__('closed_loop_controller_node')

        # localize_only: estimate the pose from tags, never move the motors
        self.localize_only = self.declare_parameter('localize_only', False).value

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

        # pose estimate: x, y, theta and its covariance
        self.lock = threading.Lock()
        self.state = np.zeros(3)
        self.cov = np.diag([0.02**2, 0.02**2, math.radians(3)**2])
        self.initialized = False
        self.rejects = 0
        self.total_rejects = 0
        self.speed = 0.0
        self.turn_rate = 0.0
        self.ignore_until = 0.0
        self.last_accept_time = 0.0
        self.last_tag_time = 0.0
        self.last_tag = ''
        self.last_predict = time.monotonic()
        self.tick = 0

        waypoints = self.load_waypoints()
        if not self.localize_only:
            self.state = np.array(waypoints[0])
            self.initialized = True

        # log the pose for the trajectory plot
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        self.pose_file = open(os.path.join(LOG_DIR, f'pose_{stamp}.csv'), 'w', newline='')
        self.pose_log = csv.writer(self.pose_file)
        self.pose_log.writerow(['t', 'x', 'y', 'theta', 'L', 'R', 'last_tag', 'tag_age'])
        self.waypoint_file = open(os.path.join(LOG_DIR, f'waypoints_{stamp}.csv'), 'w', newline='')
        self.waypoint_log = csv.writer(self.waypoint_file)
        self.waypoint_log.writerow(['i', 'goal_x', 'goal_y', 'goal_theta', 'x', 'y', 'theta', 'err_m', 'err_rad'])
        self.start_time = time.monotonic()

        self.subscription = self.create_subscription(TFMessage, '/tf', self.on_tf, 50)

        # setup timer to publish commands at 20Hz
        self.timer = self.create_timer(0.05, self.publish_motor_commands)  # 20Hz

        self.get_logger().info('Closed Loop Controller Node started')

        # follow waypoints in a separate thread (timer keeps publishing)
        if not self.localize_only:
            self.waypoint_thread = threading.Thread(target=self.follow_waypoints, args=(waypoints,), daemon=True)
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

    # ---- pose estimate ----

    def on_tf(self, msg):
        """correct the pose estimate with every tag the camera sees"""
        for t in msg.transforms:
            if t.child_frame_id not in TAGS:
                continue
            p = t.transform.translation
            q = t.transform.rotation
            # tag +z points out of its face: third column of the rotation matrix
            nx = 2 * (q.x * q.z + q.w * q.y)
            nz = 1 - 2 * (q.x * q.x + q.y * q.y)
            # camera: x right, z forward. robot: x forward, y left.
            rng = math.hypot(p.x, p.z)
            bearing = math.atan2(-p.x, p.z)
            tag_angle = math.atan2(-nx, nz)
            with self.lock:
                # a reading taken mid-spin is too old to match the current heading
                spinning = self.current_L * self.current_R < 0
                if not spinning and time.monotonic() >= self.ignore_until:
                    self.correct(TAGS[t.child_frame_id], rng, bearing, tag_angle)
                self.last_tag_time = time.monotonic()
                self.last_tag = t.child_frame_id

    def pose_from_tag(self, tag, rng, bearing, tag_angle):
        """the robot pose one tag reading implies by itself"""
        theta = wrap(tag[2] - tag_angle)
        cam_x = tag[0] - rng * math.cos(theta + bearing)
        cam_y = tag[1] - rng * math.sin(theta + bearing)
        return np.array([cam_x - CAMERA_FORWARD * math.cos(theta),
                         cam_y - CAMERA_FORWARD * math.sin(theta), theta])

    def correct(self, tag, rng, bearing, tag_angle):
        """extended Kalman filter update from one tag reading"""
        if not self.initialized:
            self.state = self.pose_from_tag(tag, rng, bearing, tag_angle)
            self.initialized = True
            return

        x, y, theta = self.state
        s, c = math.sin(theta), math.cos(theta)
        dx = tag[0] - (x + CAMERA_FORWARD * c)
        dy = tag[1] - (y + CAMERA_FORWARD * s)
        r2 = max(dx * dx + dy * dy, 1e-6)
        r = math.sqrt(r2)

        innovation = np.array([rng - r,
                               wrap(bearing - (math.atan2(dy, dx) - theta)),
                               wrap(tag_angle - (tag[2] - theta))])
        f = CAMERA_FORWARD
        H = np.array([
            [-dx / r, -dy / r, (dx * f * s - dy * f * c) / r],
            [dy / r2, -dx / r2, (-dy * f * s - dx * f * c) / r2 - 1.0],
            [0.0, 0.0, -1.0],
        ])
        # a far tag gives good bearing but poor direction, so trust it less
        # while moving, the delay adds range error too
        R = np.diag([(RANGE_STD + RANGE_STD_PER_M * r)**2 + (self.speed * TAG_DELAY)**2,
                     BEARING_STD**2 + (self.turn_rate * TAG_DELAY)**2,
                     (TAG_ANGLE_STD + TAG_ANGLE_STD_PER_M * r)**2])
        S = H @ self.cov @ H.T + R

        # ignore readings that disagree too much with the estimate
        if innovation @ np.linalg.solve(S, innovation) > GATE:
            self.rejects += 1
            self.total_rejects += 1
            if self.rejects >= MAX_REJECTS:
                # let the next readings pull the estimate over instead of jumping it
                self.get_logger().warn('Tag readings keep disagreeing: trusting the estimate less')
                self.cov = self.cov + np.diag([REJECT_POS_STD**2, REJECT_POS_STD**2, REJECT_ANGLE_STD**2])
                self.rejects = 0
            return
        self.rejects = 0
        self.last_accept_time = time.monotonic()

        K = self.cov @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ innovation
        self.state[2] = wrap(self.state[2])
        self.cov = (np.eye(3) - K @ H) @ self.cov

    def predict(self, dt):
        """move the estimate using the current motor commands (no tag needed)"""
        L, R = self.current_L, self.current_R
        if L * R < 0:   # spinning in place
            v = 0.0
            speed = ANGULAR_SPEED_LEFT if R > L else ANGULAR_SPEED_RIGHT
            w = speed * math.copysign(1.0, R - L) * abs(R - L) / (2 * TURN_SPEED)
        else:           # driving: DRIVE_R - DRIVE_L is trim, not steering
            v = LINEAR_SPEED * (L + R) / (DRIVE_L + DRIVE_R)
            w = ANGULAR_SPEED_LEFT * ((R - L) - (DRIVE_R - DRIVE_L) * (L + R) / (DRIVE_L + DRIVE_R)) / (2 * TURN_SPEED)

        with self.lock:
            self.speed = v
            self.turn_rate = w
            theta = self.state[2]
            self.state[0] += v * math.cos(theta) * dt
            self.state[1] += v * math.sin(theta) * dt
            self.state[2] = wrap(theta + w * dt)
            F = np.array([[1.0, 0.0, -v * math.sin(theta) * dt],
                          [0.0, 1.0, v * math.cos(theta) * dt],
                          [0.0, 0.0, 1.0]])
            # uncertainty grows with distance driven, so far tags can correct drift early
            moved = abs(v) * dt
            Q = np.diag([DRIVE_POS_STD**2 * moved + 1e-7,
                         DRIVE_POS_STD**2 * moved + 1e-7,
                         DRIVE_ANGLE_STD**2 * moved + (0.15 * abs(w) * dt)**2 + 1e-7])
            self.cov = F @ self.cov @ F.T + Q

    def pose(self):
        with self.lock:
            return tuple(self.state)

    def tag_age(self):
        with self.lock:
            return time.monotonic() - self.last_tag_time

    # ---- control ----

    def stop(self):
        if self.current_L * self.current_R < 0:
            self.ignore_until = time.monotonic() + SPIN_SETTLE
        self.current_L = 0.0
        self.current_R = 0.0

    def stop_and_look(self):
        """stop so the camera gets sharp images, return True if it sees a tag"""
        self.stop()
        time.sleep(PAUSE)
        return self.tag_age() < PAUSE

    def spin(self, angle):
        """spin toward angle (positive = left) for the time it should take"""
        speed = ANGULAR_SPEED_LEFT if angle > 0 else ANGULAR_SPEED_RIGHT
        direction = 1.0 if angle > 0 else -1.0
        self.current_L = -direction * TURN_SPEED
        self.current_R = direction * TURN_SPEED
        time.sleep(max(abs(angle) / speed, MIN_PULSE))
        self.stop()
        # timing errors add up over a spin, so the next tag reading must be able to fix them
        with self.lock:
            self.cov[2, 2] += (SPIN_UNCERTAINTY * angle)**2

    def wait_for_reading(self, since):
        """wait until a tag reading newer than since has corrected the estimate"""
        end = time.monotonic() + FRESH_TAG_WAIT
        while self.running and time.monotonic() < end:
            if self.last_accept_time > since:
                return True
            time.sleep(0.05)
        return False

    def search_for_tag(self):
        """spin in small steps until a tag appears, return False if none does"""
        self.get_logger().warn('No tag visible: searching')
        for _ in range(SEARCH_TRIES):
            if not self.running:
                return False
            self.current_L = -TURN_SPEED
            self.current_R = TURN_SPEED
            time.sleep(SEARCH_PULSE)
            if self.stop_and_look():
                self.get_logger().info(f'Found {self.last_tag}')
                return True
        return False

    def turn_to(self, heading):
        """spin to heading, re-reading the tags after each spin"""
        for _ in range(MAX_TURN_TRIES):
            if not self.running:
                return False
            error = wrap(heading - self.pose()[2])
            if abs(error) < ANGLE_TOL:
                return True
            self.spin(error)
            spin_end = time.monotonic() + SPIN_SETTLE
            if not self.stop_and_look() and not self.search_for_tag():
                return False
            self.wait_for_reading(spin_end)
        self.get_logger().warn(f'Heading still off after {MAX_TURN_TRIES} tries')
        return True

    def drive_to(self, goal_x, goal_y):
        """drive along the line to a point, correcting with left and right speed"""
        start_x, start_y, _ = self.pose()
        leg_x, leg_y = goal_x - start_x, goal_y - start_y
        leg_heading = math.atan2(leg_y, leg_x)
        leg_length = max(math.hypot(leg_x, leg_y), 1e-6)
        while self.running:
            x, y, theta = self.pose()
            dx, dy = goal_x - x, goal_y - y
            distance = math.hypot(dx, dy)
            # reached it, or went past it along the leg
            if distance < POS_TOL or dx * leg_x + dy * leg_y < 0:
                break

            # no tag for too long: stop and look before going on
            if self.tag_age() > TAG_TIMEOUT:
                if not self.stop_and_look() and not self.search_for_tag():
                    return False
                continue

            # how far left of the line the robot is, and the heading that returns to it
            off_line = (leg_x * (y - start_y) - leg_y * (x - start_x)) / leg_length
            error = wrap(leg_heading - math.atan(LINE_GAIN * off_line) - theta)
            if abs(error) > MAX_HEADING_ERROR and distance > SLOW_DIST:
                self.stop()
                if not self.turn_to(math.atan2(dy, dx)):
                    return False
                continue

            steer = max(-MAX_STEER, min(MAX_STEER, HEADING_GAIN * error))
            factor = max(MIN_SPEED_FACTOR, min(1.0, distance / SLOW_DIST))
            self.current_L = max(0.0, DRIVE_L * factor - steer)
            self.current_R = max(0.0, DRIVE_R * factor + steer)
            time.sleep(0.05)
        self.stop()
        return self.running

    def refine_position(self, goal_x, goal_y):
        """after stopping, creep closer if a fresh tag reading says the stop is off"""
        for _ in range(REFINE_TRIES):
            if not self.running:
                return False
            since = time.monotonic()
            self.stop_and_look()
            # without a fresh reading the error is not known well enough to act on
            if not self.wait_for_reading(since):
                self.get_logger().warn('No fresh tag reading, skipping the position correction')
                return True
            x, y, theta = self.pose()
            dx, dy = goal_x - x, goal_y - y
            distance = math.hypot(dx, dy)
            if distance < REFINE_TOL:
                return True
            self.get_logger().info(f'Stopped {distance*100:.1f} cm off, correcting')

            # overshot: back up instead of spinning around next to the tag
            bearing = math.atan2(dy, dx)
            backward = abs(wrap(bearing - theta)) > math.pi / 2
            if not self.turn_to(wrap(bearing + math.pi) if backward else bearing):
                return False
            sign = -1.0 if backward else 1.0
            end = time.monotonic() + CREEP_TIMEOUT
            while self.running and time.monotonic() < end:
                x, y, _ = self.pose()
                dx, dy = goal_x - x, goal_y - y
                # reached it, or went past it
                if math.hypot(dx, dy) < POS_TOL or dx * math.cos(bearing) + dy * math.sin(bearing) < 0:
                    break
                self.current_L = sign * DRIVE_L * MIN_SPEED_FACTOR
                self.current_R = sign * DRIVE_R * MIN_SPEED_FACTOR
                time.sleep(0.05)
            self.stop()
        return True

    def record_waypoint(self, i, goal):
        x, y, theta = self.pose()
        err_m = math.hypot(goal[0] - x, goal[1] - y)
        err_rad = wrap(goal[2] - theta)
        self.waypoint_log.writerow([i, goal[0], goal[1], goal[2], x, y, theta, err_m, err_rad])
        self.waypoint_file.flush()
        self.get_logger().info(f'Waypoint {i} done: estimate x={x:.3f} y={y:.3f} '
                               f'angle={theta:.3f}, error {err_m*100:.1f} cm, {math.degrees(err_rad):.1f} deg')

    def follow_waypoints(self, waypoints):
        """closed loop: turn toward each waypoint, drive to it, turn to its angle"""
        self.get_logger().info(f'Loaded {len(waypoints)} waypoints from {WAYPOINTS_FILE}')
        time.sleep(1.0)  # give motor_control and the camera time to connect

        # do not start blind
        if not self.stop_and_look() and not self.search_for_tag():
            self.finish('No tag visible at the start, not moving')
            return

        for i, (wx, wy, wtheta) in enumerate(waypoints[1:], start=1):
            if not self.running:
                return
            self.get_logger().info(f'Waypoint {i}: x={wx}, y={wy}, angle={wtheta}')
            x, y, _ = self.pose()
            if math.hypot(wx - x, wy - y) >= POS_TOL:
                if (not self.turn_to(math.atan2(wy - y, wx - x)) or not self.drive_to(wx, wy)
                        or not self.refine_position(wx, wy)):
                    self.finish(f'Lost on the way to waypoint {i}')
                    return
            if not self.turn_to(wtheta):
                self.finish(f'Lost at waypoint {i}')
                return
            self.stop_and_look()
            self.record_waypoint(i, (wx, wy, wtheta))

        self.finish('All waypoints reached')

    def finish(self, reason):
        self.stop()
        self.get_logger().info(f'{reason}, {self.total_rejects} tag readings rejected, shutting down...')
        self.running = False
        rclpy.shutdown()

    def publish_motor_commands(self):
        """Publish motor commands to ROS topic"""
        if self.running and rclpy.ok():
            try:
                now = time.monotonic()
                self.predict(now - self.last_predict)
                self.last_predict = now

                msg = Float32MultiArray()
                msg.data = [float(self.current_L), float(self.current_R)]
                self.publisher.publish(msg)

                # 2Hz: log the pose for the trajectory plot
                self.tick += 1
                if self.tick % 10 == 0:
                    x, y, theta = self.pose()
                    age = self.tag_age()
                    self.pose_log.writerow([round(now - self.start_time, 3), x, y, theta,
                                            self.current_L, self.current_R, self.last_tag, round(age, 3)])
                    self.pose_file.flush()
                    if self.localize_only:
                        self.get_logger().info(f'x={x:.3f} y={y:.3f} angle={theta:.3f} '
                                               f'({math.degrees(theta):.1f} deg) {self.last_tag} age={age:.2f}s')
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

        self.pose_file.close()
        self.waypoint_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ClosedLoopControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
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

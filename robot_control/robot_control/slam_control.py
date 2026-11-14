import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from tf2_msgs.msg import TFMessage
import numpy as np
import threading
import csv, math, os, re, time

LOG_DIR = os.path.expanduser('~/HW3/logs')
CAMERA_FORWARD = 0.09   # m the camera sits ahead of the robot center

# Motor commands, same values as the W and A/D keys in keyboard_control.py
DRIVE_L, DRIVE_R = 0.3, 0.25    # W: forward
TURN_SPEED = 0.3                # A/D: spin in place (L=-0.3,R=0.3 is left)

# CALIBRATE THESE on the floor: they carry the estimate when no known tag is visible
LINEAR_SPEED = 0.40         # m/s the robot actually moves with DRIVE_L/DRIVE_R
ANGULAR_SPEED_LEFT = 1.65   # rad/s spinning LEFT (HW2 used 1.4; its left turns overshot about 19%)
ANGULAR_SPEED_RIGHT = 1.72  # rad/s spinning RIGHT

# Controller (same as HW2)
POS_TOL = 0.03
ANGLE_TOL = math.radians(3)
MAX_HEADING_ERROR = math.radians(25)
HEADING_GAIN = 0.4
LINE_GAIN = 2.0
MAX_STEER = 0.10
SLOW_DIST = 0.30                # m from the corner where the robot slows down
MIN_SPEED_FACTOR = 0.5
MIN_PULSE = 0.06
MAX_TURN_TRIES = 8
PAUSE = 0.7
SPIN_SETTLE = 0.3
SPIN_UNCERTAINTY = 0.15            # a timed spin can be off by this fraction of its angle
FRESH_TAG_WAIT = 1.5

# When no tag is visible
TAG_TIMEOUT = 2.0
SEARCH_PULSE = 0.15
SEARCH_TRIES = 40

# Motion noise: standard deviation after 1 m driven
DRIVE_POS_STD = 0.05            # m along the direction of travel
DRIVE_ANGLE_STD = 0.12          # rad
TURN_RATE_STD = 0.15            # fraction of the turn rate

# Measurement noise of a tag reading (range, bearing)
RANGE_STD = 0.01
RANGE_STD_PER_M = 0.015
BEARING_STD = 0.02
TAG_DELAY = 0.25
MAX_NEW_RANGE = 2.5     # m: a farther first sighting is too rough to start a landmark
MAX_NEW_HEADING_STD = math.radians(6)   # do not start landmarks while the heading is this unsure
GATE = 13.8     # chi-square, 2 degrees of freedom, p = 0.001
MAX_REJECTS = 10
REJECT_POS_STD = 0.10
REJECT_ANGLE_STD = math.radians(8)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def make_path(shape, size, laps):
    """corners in the start frame: the robot starts at (0, 0) facing +x and turns left"""
    sides, turn = (4, math.pi / 2) if shape == 'square' else (8, math.pi / 4)
    corners = []
    x, y = 0.0, 0.0
    for _ in range(laps):
        for k in range(sides):
            x += size * math.cos(k * turn)
            y += size * math.sin(k * turn)
            corners.append((round(x, 6), round(y, 6)))
    return corners


class SlamControllerNode(Node):
    def __init__(self):
        super().__init__('slam_controller_node')

        self.shape = self.declare_parameter('path', 'square').value
        self.size = self.declare_parameter('size', 0.914 if self.shape == 'square' else 0.457).value
        self.laps = self.declare_parameter('laps', 2).value
        # map_only: build the map from tags, never move the motors
        self.map_only = self.declare_parameter('map_only', False).value

        self.publisher = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.current_L = 0.0
        self.current_R = 0.0
        self.running = True

        # state: robot x, y, theta, then x, y of each landmark in the order first seen.
        # The map frame is the robot's start pose, so the start pose is known exactly.
        self.lock = threading.Lock()
        self.state = np.zeros(3)
        self.cov = np.diag([1e-6, 1e-6, 1e-6])
        self.landmarks = {}     # tag id -> index of its x in state
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
        self.lap = 0

        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        name = f'{self.shape}_{stamp}' if not self.map_only else f'maponly_{stamp}'
        self.pose_file = open(os.path.join(LOG_DIR, f'pose_{name}.csv'), 'w', newline='')
        self.pose_log = csv.writer(self.pose_file)
        self.pose_log.writerow(['t', 'x', 'y', 'theta', 'sxx', 'syy', 'stt', 'L', 'R', 'landmarks', 'last_tag', 'tag_age', 'lap'])
        self.map_file = open(os.path.join(LOG_DIR, f'map_{name}.csv'), 'w', newline='')
        self.map_log = csv.writer(self.map_file)
        self.map_log.writerow(['lap', 'id', 'x', 'y', 'sxx', 'sxy', 'syy'])
        self.corner_file = open(os.path.join(LOG_DIR, f'corners_{name}.csv'), 'w', newline='')
        self.corner_log = csv.writer(self.corner_file)
        self.corner_log.writerow(['lap', 'i', 'goal_x', 'goal_y', 'x', 'y', 'theta', 'err_m'])
        self.start_time = time.monotonic()

        self.subscription = self.create_subscription(TFMessage, '/tf', self.on_tf, 50)
        self.timer = self.create_timer(0.05, self.publish_motor_commands)  # 20Hz
        self.get_logger().info('SLAM Controller Node started')

        if not self.map_only:
            corners = make_path(self.shape, self.size, self.laps)
            self.waypoint_thread = threading.Thread(target=self.follow_path, args=(corners,), daemon=True)
            self.waypoint_thread.start()

    # ---- EKF SLAM ----

    def on_tf(self, msg):
        """update the map and pose with every tag the camera sees"""
        for t in msg.transforms:
            m = re.search(r'(\d+)$', t.child_frame_id)
            if not m or 'tag' not in t.child_frame_id.lower() and ':' not in t.child_frame_id:
                continue
            tag_id = int(m.group(1))
            p = t.transform.translation
            # camera: x right, z forward. robot: x forward, y left.
            rng = math.hypot(p.x, p.z)
            bearing = math.atan2(-p.x, p.z)
            with self.lock:
                # a reading taken mid-spin is too old to match the current heading
                spinning = self.current_L * self.current_R < 0
                if not spinning and time.monotonic() >= self.ignore_until:
                    if tag_id in self.landmarks:
                        self.correct(tag_id, rng, bearing)
                    elif rng < MAX_NEW_RANGE and self.cov[2, 2] < MAX_NEW_HEADING_STD**2:
                        self.add_landmark(tag_id, rng, bearing)
                self.last_tag_time = time.monotonic()
                self.last_tag = f'tag_{tag_id}'

    def measurement_noise(self, r):
        # a far tag is trusted less; while moving, the reading's delay adds error too
        return np.diag([(RANGE_STD + RANGE_STD_PER_M * r)**2 + (self.speed * TAG_DELAY)**2,
                        BEARING_STD**2 + (self.turn_rate * TAG_DELAY)**2])

    def add_landmark(self, tag_id, rng, bearing):
        """first sighting: put the landmark where the reading says, with matching uncertainty"""
        x, y, theta = self.state[:3]
        a = theta + bearing
        mx = x + CAMERA_FORWARD * math.cos(theta) + rng * math.cos(a)
        my = y + CAMERA_FORWARD * math.sin(theta) + rng * math.sin(a)
        # Jacobians of the landmark position: Gx wrt robot pose, Gz wrt (range, bearing)
        Gx = np.array([[1.0, 0.0, -CAMERA_FORWARD * math.sin(theta) - rng * math.sin(a)],
                       [0.0, 1.0, CAMERA_FORWARD * math.cos(theta) + rng * math.cos(a)]])
        Gz = np.array([[math.cos(a), -rng * math.sin(a)],
                       [math.sin(a), rng * math.cos(a)]])
        n = len(self.state)
        P_rr = self.cov[:3, :3]
        P_new = Gx @ P_rr @ Gx.T + Gz @ self.measurement_noise(rng) @ Gz.T
        P_cross = Gx @ self.cov[:3, :]      # new landmark vs everything already in the state

        cov = np.zeros((n + 2, n + 2))
        cov[:n, :n] = self.cov
        cov[n:, :n] = P_cross
        cov[:n, n:] = P_cross.T
        cov[n:, n:] = P_new
        self.cov = cov
        self.state = np.append(self.state, [mx, my])
        self.landmarks[tag_id] = n
        self.get_logger().info(f'New landmark tag_{tag_id} at x={mx:.2f} y={my:.2f} '
                               f'({len(self.landmarks)} landmarks)')

    def correct(self, tag_id, rng, bearing):
        """EKF update from one reading of a landmark already in the map"""
        j = self.landmarks[tag_id]
        x, y, theta = self.state[:3]
        mx, my = self.state[j:j + 2]
        s, c = math.sin(theta), math.cos(theta)
        f = CAMERA_FORWARD
        dx = mx - (x + f * c)
        dy = my - (y + f * s)
        r2 = max(dx * dx + dy * dy, 1e-6)
        r = math.sqrt(r2)

        innovation = np.array([rng - r, wrap(bearing - (math.atan2(dy, dx) - theta))])
        H = np.zeros((2, len(self.state)))
        H[:, :3] = [[-dx / r, -dy / r, (dx * f * s - dy * f * c) / r],
                    [dy / r2, -dx / r2, (-dy * f * s - dx * f * c) / r2 - 1.0]]
        H[:, j:j + 2] = [[dx / r, dy / r],
                         [-dy / r2, dx / r2]]
        R = self.measurement_noise(r)
        S = H @ self.cov @ H.T + R

        # ignore readings that disagree too much with the estimate
        if innovation @ np.linalg.solve(S, innovation) > GATE:
            self.rejects += 1
            self.total_rejects += 1
            if self.rejects >= MAX_REJECTS:
                # let the next readings pull the robot estimate over instead of ignoring them all
                self.get_logger().warn('Tag readings keep disagreeing: trusting the robot pose less')
                self.cov[:3, :3] += np.diag([REJECT_POS_STD**2, REJECT_POS_STD**2, REJECT_ANGLE_STD**2])
                self.rejects = 0
            return
        self.rejects = 0
        self.last_accept_time = time.monotonic()

        K = self.cov @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ innovation
        self.state[2] = wrap(self.state[2])
        # Joseph form keeps the covariance symmetric and positive
        IKH = np.eye(len(self.state)) - K @ H
        self.cov = IKH @ self.cov @ IKH.T + K @ R @ K.T

    def predict(self, dt):
        """move the robot estimate using the current motor commands; landmarks do not move"""
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

            n = len(self.state)
            F = np.eye(n)       # system matrix: only the robot rows change
            F[0, 2] = -v * math.sin(theta) * dt
            F[1, 2] = v * math.cos(theta) * dt
            G = np.zeros((n, 2))    # control matrix for u = (v, w)
            G[0, 0] = math.cos(theta) * dt
            G[1, 0] = math.sin(theta) * dt
            G[2, 1] = dt
            # control noise grows with distance driven and angle turned
            moved = abs(v) * dt
            dt2 = max(dt, 1e-3)**2
            M = np.diag([(DRIVE_POS_STD**2 * moved) / dt2,
                         (DRIVE_ANGLE_STD**2 * moved + (TURN_RATE_STD * abs(w) * dt)**2) / dt2])
            Q = G @ M @ G.T
            Q[:3, :3] += np.eye(3) * 1e-8
            self.cov = F @ self.cov @ F.T + Q

    def pose(self):
        with self.lock:
            return tuple(self.state[:3])

    def tag_age(self):
        with self.lock:
            return time.monotonic() - self.last_tag_time

    def write_map(self, lap):
        with self.lock:
            for tag_id, j in sorted(self.landmarks.items()):
                P = self.cov[j:j + 2, j:j + 2]
                self.map_log.writerow([lap, tag_id, self.state[j], self.state[j + 1], P[0, 0], P[0, 1], P[1, 1]])
        self.map_file.flush()

    # ---- control (same as HW2) ----

    def stop(self):
        if self.current_L * self.current_R < 0:
            self.ignore_until = time.monotonic() + SPIN_SETTLE
        self.current_L = 0.0
        self.current_R = 0.0

    def stop_and_look(self):
        self.stop()
        time.sleep(PAUSE)
        return self.tag_age() < PAUSE

    def spin(self, angle):
        speed = ANGULAR_SPEED_LEFT if angle > 0 else ANGULAR_SPEED_RIGHT
        direction = 1.0 if angle > 0 else -1.0
        self.current_L = -direction * TURN_SPEED
        self.current_R = direction * TURN_SPEED
        time.sleep(max(abs(angle) / speed, MIN_PULSE))
        self.stop()
        with self.lock:
            self.cov[2, 2] += (SPIN_UNCERTAINTY * angle)**2

    def wait_for_reading(self, since):
        end = time.monotonic() + FRESH_TAG_WAIT
        while self.running and time.monotonic() < end:
            if self.last_accept_time > since:
                return True
            time.sleep(0.05)
        return False

    def search_for_tag(self):
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
        start_x, start_y, _ = self.pose()
        leg_x, leg_y = goal_x - start_x, goal_y - start_y
        leg_heading = math.atan2(leg_y, leg_x)
        leg_length = max(math.hypot(leg_x, leg_y), 1e-6)
        while self.running:
            x, y, theta = self.pose()
            dx, dy = goal_x - x, goal_y - y
            distance = math.hypot(dx, dy)
            if distance < POS_TOL or dx * leg_x + dy * leg_y < 0:
                break

            if self.tag_age() > TAG_TIMEOUT:
                if not self.stop_and_look() and not self.search_for_tag():
                    return False
                continue

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

    def follow_path(self, corners):
        """closed loop on the SLAM pose: turn toward each corner and drive to it"""
        per_lap = len(corners) // self.laps
        self.get_logger().info(f'{self.shape}: {self.laps} laps of {per_lap} sides, {self.size:.3f} m each')
        time.sleep(1.0)  # give motor_control and the camera time to connect

        # look before moving, so the first landmarks are in the map
        self.stop_and_look()
        self.write_map(0)

        for i, (gx, gy) in enumerate(corners, start=1):
            if not self.running:
                return
            self.lap = (i - 1) // per_lap + 1
            x, y, _ = self.pose()
            if not self.turn_to(math.atan2(gy - y, gx - x)) or not self.drive_to(gx, gy):
                self.finish(f'Lost on the way to corner {i}')
                return
            self.stop_and_look()
            x, y, theta = self.pose()
            lap = (i - 1) // per_lap + 1
            self.corner_log.writerow([lap, i, gx, gy, x, y, theta, math.hypot(gx - x, gy - y)])
            self.corner_file.flush()
            self.get_logger().info(f'Corner {i} (lap {lap}): x={x:.3f} y={y:.3f} angle={theta:.3f}, '
                                   f'{len(self.landmarks)} landmarks')
            if i % per_lap == 0:
                self.write_map(lap)

        self.turn_to(0.0)
        self.stop_and_look()
        self.finish('Path done')

    def finish(self, reason):
        self.stop()
        self.write_map(-1)
        with self.lock:
            summary = ', '.join(f'tag_{k}=({self.state[j]:.2f}, {self.state[j + 1]:.2f})'
                                for k, j in sorted(self.landmarks.items()))
        self.get_logger().info(f'{reason}, {self.total_rejects} tag readings rejected. Map: {summary}')
        self.running = False
        rclpy.shutdown()

    def publish_motor_commands(self):
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
                    with self.lock:
                        x, y, theta = self.state[:3]
                        sxx, syy, stt = self.cov[0, 0], self.cov[1, 1], self.cov[2, 2]
                        count = len(self.landmarks)
                    age = time.monotonic() - self.last_tag_time
                    self.pose_log.writerow([round(now - self.start_time, 3), x, y, theta, sxx, syy, stt,
                                            self.current_L, self.current_R, count, self.last_tag, round(age, 3), self.lap])
                    self.pose_file.flush()
                    if self.map_only and self.tick % 40 == 0:
                        self.get_logger().info(f'pose x={x:.3f} y={y:.3f} angle={math.degrees(theta):.1f} deg, '
                                               f'{count} landmarks, last {self.last_tag} age={age:.2f}s')
            except Exception as e:
                if self.running:
                    self.get_logger().error(f'Error publishing command: {str(e)}')

    def destroy_node(self):
        if self.running:
            self.write_map(-1)
        self.running = False
        msg = Float32MultiArray()
        msg.data = [0.0, 0.0]
        self.publisher.publish(msg)
        self.get_logger().info('Stop command sent')
        self.pose_file.close()
        self.map_file.close()
        self.corner_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SlamControllerNode()
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
        return 0


if __name__ == '__main__':
    main()

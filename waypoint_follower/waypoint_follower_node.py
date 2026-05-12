"""Waypoint follower node.

Buffers PoseStamped waypoints arriving on a topic and drives the robot through
them by publishing geometry_msgs/TwistStamped to cmd_vel using a simple
follow-the-carrot rule. Each odometry message triggers one control tick
(compute heading error to the front-of-buffer waypoint, publish a TwistStamped),
so the command rate matches the odom rate and the command always uses the
freshest pose.
"""

import math
from collections import deque
from typing import Deque, Optional

import rclpy
from geometry_msgs.msg import Pose, PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time


def _best_effort(depth: int) -> QoSProfile:
    return QoSProfile(
        depth=depth,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
    )


# Per-topic QoS — all BEST_EFFORT, queue sized for the topic's role.
ODOM_QOS = _best_effort(10)
CMD_VEL_QOS = _best_effort(10)
WAYPOINT_QOS = _best_effort(200)
PATH_QOS = _best_effort(10)


def euclidean_distance(a: Pose, b: Pose) -> float:
    return math.hypot(
        a.position.x - b.position.x,
        a.position.y - b.position.y,
    )


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def transform_base_to_odom(pose_in_base: Pose, odom_pose: Pose) -> Pose:
    """Compose T_odom_base · pose_in_base, planar (yaw-only) rotation.

    Treats the incoming pose as expressed in base_link (camera ≡ base_link by
    project assumption) and lifts it into the odometry frame using the most
    recent odometry pose. Position is rotated by odom yaw and translated by
    odom xyz; orientation is composed as yaw addition (matches the controller's
    planar model — full 3D quaternion composition is unnecessary here).
    """
    yaw = yaw_from_quaternion(odom_pose.orientation)
    c, s = math.cos(yaw), math.sin(yaw)
    out = Pose()
    out.position.x = odom_pose.position.x + c * pose_in_base.position.x - s * pose_in_base.position.y
    out.position.y = odom_pose.position.y + s * pose_in_base.position.x + c * pose_in_base.position.y
    out.position.z = odom_pose.position.z + pose_in_base.position.z
    total_yaw = yaw + yaw_from_quaternion(pose_in_base.orientation)
    out.orientation.x = 0.0
    out.orientation.y = 0.0
    out.orientation.z = math.sin(total_yaw / 2.0)
    out.orientation.w = math.cos(total_yaw / 2.0)
    return out


class WaypointFollower(Node):
    def __init__(self) -> None:
        super().__init__('waypoint_follower')

        self.declare_parameter('waypoint_topic', 'waypoint')
        self.declare_parameter('odom_topic', 'odom')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('cmd_vel_frame_id', 'base_link')
        self.declare_parameter('path_topic', 'waypoint_path')
        self.declare_parameter('goal_tolerance', 0.5)
        self.declare_parameter('desired_linear_velocity', 3.0)
        self.declare_parameter('max_angular_velocity', 1.0)
        self.declare_parameter('angular_gain', 0.5)
        # Buffer down-sampling: drop incoming waypoints closer than this to
        # the previously accepted one. 0 disables the filter.
        self.declare_parameter('waypoint_spacing', 0.5)
        # Velocity-from-timestamps: derive segment speed from waypoint stamps.
        self.declare_parameter('use_waypoint_timestamps', True)
        self.declare_parameter('min_linear_velocity', 0.05)
        self.declare_parameter('max_linear_velocity', 0.8)
        # Pause motion until the integrated path length from the last reached
        # waypoint through the buffer is at least this large. 0 disables.
        self.declare_parameter('minimal_following_distance', 0.0)

        self.waypoint_topic = self.get_parameter('waypoint_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.cmd_vel_frame_id = self.get_parameter('cmd_vel_frame_id').value
        self.path_topic = self.get_parameter('path_topic').value
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.desired_linear_velocity = float(
            self.get_parameter('desired_linear_velocity').value)
        self.max_angular_velocity = float(self.get_parameter('max_angular_velocity').value)
        self.angular_gain = float(self.get_parameter('angular_gain').value)
        self.waypoint_spacing = float(self.get_parameter('waypoint_spacing').value)
        self.use_waypoint_timestamps = bool(
            self.get_parameter('use_waypoint_timestamps').value)
        self.min_linear_velocity = float(self.get_parameter('min_linear_velocity').value)
        self.max_linear_velocity = float(self.get_parameter('max_linear_velocity').value)
        self.minimal_following_distance = float(
            self.get_parameter('minimal_following_distance').value)

        self.waypoint_buffer: Deque[PoseStamped] = deque()
        # Position of the last waypoint we accepted into the buffer (kept
        # across pops so spacing stays consistent as waypoints stream in).
        self._last_accepted_wp_pose: Optional[Pose] = None
        self._first_odom_seen = False
        # Reference for segment-velocity computation: the last reached waypoint,
        # or the first odom we saw before any waypoint was reached.
        self._segment_start_pose: Optional[Pose] = None
        self._segment_start_time: Optional[Time] = None
        # Path length from _segment_start_pose (or buffer[0] if no odom yet)
        # through the buffer to its last entry. Maintained incrementally.
        self._integrated_distance: float = 0.0
        # Latest odometry pose + frame, cached for transforming incoming
        # camera-frame waypoints into the odometry frame on receipt.
        self._latest_odom_pose: Optional[Pose] = None
        self._latest_odom_frame: str = ''

        self.create_subscription(
            PoseStamped, self.waypoint_topic, self._waypoint_callback,
            WAYPOINT_QOS)
        self.create_subscription(
            Odometry, self.odom_topic, self._odom_callback, ODOM_QOS)

        self.cmd_vel_pub = self.create_publisher(
            TwistStamped, self.cmd_vel_topic, CMD_VEL_QOS)
        self.path_pub = self.create_publisher(
            Path, self.path_topic, PATH_QOS)

        self.get_logger().info(
            f'waypoint_follower ready (BEST_EFFORT QoS on all topics, '
            f'control runs on each odom message):\n'
            f'  waypoints (PoseStamped)  <- {self.waypoint_topic}\n'
            f'  odom (Odometry)          <- {self.odom_topic}\n'
            f'  cmd_vel (TwistStamped)   -> {self.cmd_vel_topic} '
            f'(frame_id={self.cmd_vel_frame_id!r})\n'
            f'  path (Path)              -> {self.path_topic}')

    def _waypoint_callback(self, msg: PoseStamped) -> None:
        # Lift the waypoint from the camera/base frame into the odometry
        # frame using the freshest odometry snapshot. Camera ≡ base_link by
        # project assumption — no static camera-mount offset is applied.
        # All downstream logic operates on the transformed pose.
        if self._latest_odom_pose is None:
            self.get_logger().warn(
                f'[wp_cb] DROP: no odometry seen yet on {self.odom_topic}, '
                f'cannot transform waypoint into odom frame',
                throttle_duration_sec=2.0)
            return
        msg.pose = transform_base_to_odom(msg.pose, self._latest_odom_pose)
        msg.header.frame_id = self._latest_odom_frame

        # Down-sample by spacing: drop any waypoint that is closer than
        # waypoint_spacing to the last accepted one. The first waypoint
        # always passes.
        if (self.waypoint_spacing > 0.0
                and self._last_accepted_wp_pose is not None):
            if (euclidean_distance(self._last_accepted_wp_pose, msg.pose)
                    < self.waypoint_spacing):
                return
        # Extend the integrated path length before appending so buffer[-1]
        # still refers to the previous tail.
        if self.waypoint_buffer:
            self._integrated_distance += euclidean_distance(
                self.waypoint_buffer[-1].pose, msg.pose)
        elif self._segment_start_pose is not None:
            self._integrated_distance += euclidean_distance(
                self._segment_start_pose, msg.pose)
        self.waypoint_buffer.append(msg)
        self._last_accepted_wp_pose = msg.pose
        self._publish_path()

    def _odom_callback(self, msg: Odometry) -> None:
        if not self._first_odom_seen:
            self._first_odom_seen = True
            self.get_logger().info(
                f'[odom_cb] FIRST odom received on {self.odom_topic} '
                f'frame={msg.header.frame_id!r} '
                f'pos=({msg.pose.pose.position.x:.2f}, '
                f'{msg.pose.pose.position.y:.2f})')

        # Cache for the waypoint transform (camera/base → odom).
        self._latest_odom_pose = msg.pose.pose
        self._latest_odom_frame = msg.header.frame_id

        if self._segment_start_pose is None:
            self._segment_start_pose = msg.pose.pose
            self._segment_start_time = Time.from_msg(msg.header.stamp)
            # Catch up the integrated distance with the segment from the
            # newly-known start to whatever was buffered before odom arrived.
            if self.waypoint_buffer:
                self._integrated_distance += euclidean_distance(
                    self._segment_start_pose,
                    self.waypoint_buffer[0].pose)

        self._control_step(msg)

    def _publish_path(self) -> None:
        if not self.waypoint_buffer:
            return
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.waypoint_buffer[0].header.frame_id or 'map'
        path.poses = list(self.waypoint_buffer)
        self.path_pub.publish(path)

    def _publish_cmd(self, twist: Twist) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.cmd_vel_frame_id
        msg.twist = twist
        self.cmd_vel_pub.publish(msg)

    def _stop(self) -> None:
        self._publish_cmd(Twist())

    def _control_step(self, odom: Odometry) -> None:
        if not self.waypoint_buffer:
            self.get_logger().info(
                f'[ctrl] BAIL: buffer empty, waiting for waypoints on '
                f'{self.waypoint_topic}',
                throttle_duration_sec=2.0)
            return

        robot_pose = odom.pose.pose
        target = self.waypoint_buffer[0]
        distance_to_target = euclidean_distance(robot_pose, target.pose)

        # Frame sanity check — math assumes both are in the same frame.
        odom_frame = odom.header.frame_id
        target_frame = target.header.frame_id
        if odom_frame and target_frame and odom_frame != target_frame:
            self.get_logger().warn(
                f'[ctrl] FRAME MISMATCH: odom in {odom_frame!r}, '
                f'waypoint in {target_frame!r} — distances will be wrong',
                throttle_duration_sec=5.0)

        if distance_to_target < self.goal_tolerance:
            reached = self.waypoint_buffer.popleft()
            # Subtract the segment we just consumed before advancing
            # _segment_start_pose, so the invariant is preserved.
            if self._segment_start_pose is not None:
                self._integrated_distance -= euclidean_distance(
                    self._segment_start_pose, reached.pose)
                if self._integrated_distance < 0.0:
                    self._integrated_distance = 0.0
            self._segment_start_pose = reached.pose
            self._segment_start_time = Time.from_msg(reached.header.stamp)
            self.get_logger().info(
                f'[ctrl] REACHED ({reached.pose.position.x:.2f}, '
                f'{reached.pose.position.y:.2f}) dist={distance_to_target:.3f} '
                f'< tol={self.goal_tolerance}; '
                f'{len(self.waypoint_buffer)} remaining')
            self._publish_path()
            if not self.waypoint_buffer:
                self._stop()
                self.get_logger().info('[ctrl] DONE — all waypoints reached, '
                                       'sent zero TwistStamped.')
            return

        if self._integrated_distance < self.minimal_following_distance:
            self._stop()
            self.get_logger().info(
                f'[ctrl] GATED: integrated={self._integrated_distance:.2f} m '
                f'< minimal_following_distance={self.minimal_following_distance:.2f} m, '
                f'buffer={len(self.waypoint_buffer)} — holding still',
                throttle_duration_sec=1.0)
            return

        desired_v = self._segment_desired_velocity(target)
        cmd = self._compute_carrot_cmd(robot_pose, target, desired_v)

        rx = robot_pose.position.x
        ry = robot_pose.position.y
        tx = target.pose.position.x
        ty = target.pose.position.y
        ryaw = math.degrees(yaw_from_quaternion(robot_pose.orientation))
        bearing = math.degrees(math.atan2(ty - ry, tx - rx))
        heading_err = math.degrees(
            math.atan2(math.sin(math.radians(bearing - ryaw)),
                       math.cos(math.radians(bearing - ryaw))))
        # No throttle: log every control tick so the user can see the actual
        # tick-to-tick evolution of heading_err. If this is too noisy, raise
        # the throttle back up, but be aware it hides intermediate ticks.
        self.get_logger().info(
            f'[ctrl] PUB cmd_vel lin={cmd.linear.x:+.3f} ang={cmd.angular.z:+.3f} '
            f'| robot=({rx:.2f},{ry:.2f}) yaw={ryaw:+.1f}deg '
            f'| target=({tx:.2f},{ty:.2f}) bearing={bearing:+.1f}deg '
            f'heading_err={heading_err:+.1f}deg '
            f'| dist={distance_to_target:.2f} int_dist={self._integrated_distance:.2f} '
            f'desired_v={desired_v:.2f} '
            f'subs={self.cmd_vel_pub.get_subscription_count()}')

        self._publish_cmd(cmd)

    def _segment_desired_velocity(self, target: PoseStamped) -> float:
        """Linear speed for this segment, derived from time delta between
        waypoint stamps. Falls back to ``desired_linear_velocity`` if
        timestamps are missing or non-monotonic, or if the feature is off.
        """
        if not self.use_waypoint_timestamps:
            return self.desired_linear_velocity
        if self._segment_start_pose is None or self._segment_start_time is None:
            return self.desired_linear_velocity

        target_time = Time.from_msg(target.header.stamp)
        dt_ns = (target_time - self._segment_start_time).nanoseconds
        if dt_ns <= 0:
            return self.desired_linear_velocity

        seg_dist = math.hypot(
            target.pose.position.x - self._segment_start_pose.position.x,
            target.pose.position.y - self._segment_start_pose.position.y,
        )
        if seg_dist < 1e-6:
            return self.desired_linear_velocity

        v = seg_dist / (dt_ns * 1e-9)
        return max(self.min_linear_velocity, min(self.max_linear_velocity, v))

    def _compute_carrot_cmd(
        self,
        robot_pose: Pose,
        target: PoseStamped,
        desired_linear_velocity: float,
    ) -> Twist:
        """Forward-only follow-the-carrot.

        heading_error = atan2(sin(bearing - yaw), cos(bearing - yaw)) in (-pi, pi].
        linear.x = desired_v * max(0, cos(heading_error)) — smooth roll-off.
        angular.z = clamp(angular_gain * heading_error, ±max_angular_velocity).

        Assumes the inter-waypoint heading change stays well under 90° so the
        cosine roll-off never bites. If |heading_error| ever exceeds 90°,
        that indicates a data problem (sim odometry inconsistent with
        waypoint trajectory), not a controller problem.
        """
        ryaw = yaw_from_quaternion(robot_pose.orientation)
        bearing = math.atan2(
            target.pose.position.y - robot_pose.position.y,
            target.pose.position.x - robot_pose.position.x,
        )
        heading_error = -math.atan2(
            math.sin(bearing - ryaw), math.cos(bearing - ryaw))

        cmd = Twist()
        cmd.linear.x = desired_linear_velocity * max(0.0, math.cos(heading_error))
        cmd.angular.z = max(
            -self.max_angular_velocity,
            min(self.max_angular_velocity, self.angular_gain * heading_error),
        )
        return cmd


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WaypointFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

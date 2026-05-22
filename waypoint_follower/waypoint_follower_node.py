"""Waypoint follower node.

Two control modes, selectable via the ``mode`` parameter:

* ``waypoints`` (default) — buffers PoseStamped waypoints synthesised from
  incoming markers and drives the robot through them by publishing
  geometry_msgs/TwistStamped to cmd_vel using a simple follow-the-carrot rule.
* ``carrot`` — locks on to a single marker and follows it like a carrot,
  maintaining ``desired_carrot_distance`` at all times (never backs up). If
  the marker disappears for longer than ``marker_timeout``, drives to the
  last known marker position, rotates to the last known marker yaw, and
  stops.

In either mode, each odometry message triggers one control tick (compute the
command, publish a TwistStamped), so the command rate matches the odom rate
and the command always uses the freshest pose.
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
from whycode_interfaces.msg import MarkerArray


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

        # Mode: 'waypoints' (buffer markers as waypoints, follow them) or
        # 'carrot' (lock on to one marker, follow it at a fixed standoff).
        self.declare_parameter('mode', 'waypoints')

        self.declare_parameter('marker_topic', '/whycode_node/markers')
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

        # Carrot-mode parameters.
        # Standoff distance to maintain from the marker; never drive closer
        # than this and never back up if already inside it.
        self.declare_parameter('desired_carrot_distance', 1.5)
        # Time without any marker detection before the marker is treated as
        # lost and we switch to "go to last known position" behaviour.
        self.declare_parameter('marker_timeout', 1.0)
        # Yaw tolerance (rad) when aligning to the last known marker yaw
        # after the marker is lost — below this we declare the manoeuvre done.
        self.declare_parameter('final_yaw_tolerance', 0.1)
        # Linear velocity slew limits for the carrot mode (m/s^2). ``decel``
        # also drives the physics-correct stopping profile near the standoff
        # (v_target = sqrt(2 * decel * gap)).
        self.declare_parameter('linear_accel', 0.5)
        self.declare_parameter('linear_decel', 0.3)

        self.mode = str(self.get_parameter('mode').value).lower()
        if self.mode not in ('waypoints', 'carrot'):
            self.get_logger().warn(
                f"unknown mode={self.mode!r} — falling back to 'waypoints'")
            self.mode = 'waypoints'

        self.marker_topic = self.get_parameter('marker_topic').value
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
        self.desired_carrot_distance = float(
            self.get_parameter('desired_carrot_distance').value)
        self.marker_timeout = float(self.get_parameter('marker_timeout').value)
        self.final_yaw_tolerance = float(
            self.get_parameter('final_yaw_tolerance').value)
        self.linear_accel = max(1e-3, float(self.get_parameter('linear_accel').value))
        self.linear_decel = max(1e-3, float(self.get_parameter('linear_decel').value))

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

        # Carrot-mode state: the most recent marker pose in the odom frame,
        # its yaw, and the wall-clock time of the last detection. ``None``
        # until the first marker is seen.
        self._carrot_pose: Optional[Pose] = None
        self._carrot_yaw: float = 0.0
        self._carrot_last_seen: Optional[Time] = None
        # Slew-rate state for the carrot linear-velocity controller. We
        # remember the last commanded linear velocity and the time it was
        # published so we can clamp |dv/dt| to ``linear_accel`` / ``linear_decel``.
        self._prev_linear_cmd: float = 0.0
        self._prev_cmd_time: Optional[Time] = None

        self.create_subscription(
            MarkerArray, self.marker_topic, self._marker_callback,
            WAYPOINT_QOS)
        self.create_subscription(
            Odometry, self.odom_topic, self._odom_callback, ODOM_QOS)

        self.cmd_vel_pub = self.create_publisher(
            TwistStamped, self.cmd_vel_topic, CMD_VEL_QOS)
        self.path_pub = self.create_publisher(
            Path, self.path_topic, PATH_QOS)

        self.get_logger().info(
            f'waypoint_follower ready (mode={self.mode!r}, '
            f'BEST_EFFORT QoS on all topics, '
            f'control runs on each odom message):\n'
            f'  markers (MarkerArray)    <- {self.marker_topic}\n'
            f'  odom (Odometry)          <- {self.odom_topic}\n'
            f'  cmd_vel (TwistStamped)   -> {self.cmd_vel_topic} '
            f'(frame_id={self.cmd_vel_frame_id!r})\n'
            f'  path (Path)              -> {self.path_topic}')

    def _marker_callback(self, msg: MarkerArray) -> None:
        # Whycode (and our test bag-replay) publishes detected markers in the
        # camera frame, treated as base_link by project assumption. We lift
        # them into the odometry frame using the freshest odometry snapshot
        # before any other logic runs.
        if self._latest_odom_pose is None:
            self.get_logger().warn(
                f'[mk_cb] DROP: no odometry seen yet on {self.odom_topic}',
                throttle_duration_sec=2.0)
            return
        if not msg.markers:
            return  # empty detection — silent skip

        # Selection reference: the last thing we locked onto. For waypoints
        # mode that's the last accepted waypoint, for carrot mode it's the
        # current carrot pose. ``None`` means we have not locked on yet.
        if self.mode == 'carrot':
            lock_ref = self._carrot_pose
        else:
            lock_ref = self._last_accepted_wp_pose

        # Selection: at start-up we require an unambiguous single marker;
        # once tracking has begun, pick the marker closest (in odom) to the
        # last lock-on reference to stay on the same physical fiducial.
        candidates = [transform_base_to_odom(m.position, self._latest_odom_pose)
                      for m in msg.markers]
        if lock_ref is None:
            if len(candidates) != 1:
                self.get_logger().info(
                    f'[mk_cb] WAIT: {len(candidates)} markers visible at start — '
                    f'need exactly 1 to lock on',
                    throttle_duration_sec=2.0)
                return
            chosen = candidates[0]
        else:
            chosen = min(candidates,
                         key=lambda p: euclidean_distance(p, lock_ref))

        if self.mode == 'carrot':
            self._carrot_pose = chosen
            self._carrot_yaw = yaw_from_quaternion(chosen.orientation)
            self._carrot_last_seen = self.get_clock().now()
            self._publish_carrot_path()
            return

        synth = PoseStamped()
        synth.header.stamp = msg.header.stamp
        synth.header.frame_id = self._latest_odom_frame
        synth.pose = chosen
        self._buffer_waypoint(synth)

    def _buffer_waypoint(self, msg: PoseStamped) -> None:
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

        self._control_step(msg)

    def _publish_path(self) -> None:
        if not self.waypoint_buffer:
            return
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.waypoint_buffer[0].header.frame_id or 'map'
        path.poses = list(self.waypoint_buffer)
        self.path_pub.publish(path)

    def _publish_carrot_path(self) -> None:
        """Publish the current carrot as a single-pose Path (for rviz)."""
        if self._carrot_pose is None:
            return
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self._latest_odom_frame or 'map'
        pose_stamped = PoseStamped()
        pose_stamped.header = path.header
        pose_stamped.pose = self._carrot_pose
        path.poses = [pose_stamped]
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
        if self.mode == 'carrot':
            self._control_step_carrot(odom)
            return

        if not self.waypoint_buffer:
            self.get_logger().info(
                f'[ctrl] BAIL: buffer empty, waiting for markers on '
                f'{self.marker_topic}',
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

    def _carrot_target_velocity(
        self, remaining_distance: float, heading_error: float
    ) -> float:
        """Forward velocity that brings the robot to rest in
        ``remaining_distance`` metres, capped at ``desired_linear_velocity``
        and rolled off by cos(heading_error).

        Uses v = sqrt(2 * linear_decel * remaining_distance) — the largest
        speed from which we can still stop within ``remaining_distance`` at a
        constant deceleration of ``linear_decel``. This gives a smooth,
        physics-correct approach: cruise far from the goal, ramp down as we
        close in, hit zero exactly at the stop point.
        """
        if remaining_distance <= 0.0:
            return 0.0
        v_decel = math.sqrt(2.0 * self.linear_decel * remaining_distance)
        v = min(self.desired_linear_velocity, v_decel)
        return v * max(0.0, math.cos(heading_error))

    def _slew_linear(self, target_v: float, dt: float) -> float:
        """Clamp |dv/dt| to ``linear_accel`` (speeding up) or
        ``linear_decel`` (slowing down), starting from ``_prev_linear_cmd``."""
        if dt <= 0.0:
            return self._prev_linear_cmd
        if target_v > self._prev_linear_cmd:
            step = min(target_v - self._prev_linear_cmd, self.linear_accel * dt)
            return self._prev_linear_cmd + step
        step = min(self._prev_linear_cmd - target_v, self.linear_decel * dt)
        return self._prev_linear_cmd - step

    def _control_step_carrot(self, odom: Odometry) -> None:
        """Carrot mode: follow the latest marker at a fixed standoff.

        While the marker is visible, hold ``desired_carrot_distance`` (never
        drive closer, never back up). If the marker is lost for longer than
        ``marker_timeout``, drive to the last known marker position, rotate
        to the last known marker yaw, then stop.

        Linear velocity is shaped by a sqrt(2·decel·gap) profile and then
        slew-limited by ``linear_accel`` / ``linear_decel`` for smooth
        accel/decel.
        """
        if self._carrot_pose is None or self._carrot_last_seen is None:
            # No carrot yet — keep the slew state zeroed so we restart cleanly
            # the moment a marker shows up.
            self._prev_linear_cmd = 0.0
            self._prev_cmd_time = None
            self._stop()
            self.get_logger().info(
                f'[ctrl/carrot] WAIT: no marker seen yet on {self.marker_topic}',
                throttle_duration_sec=2.0)
            return

        robot_pose = odom.pose.pose
        target = self._carrot_pose
        distance_to_target = euclidean_distance(robot_pose, target)

        # Frame sanity check — math assumes both are in the same frame.
        odom_frame = odom.header.frame_id
        if (self._latest_odom_frame and odom_frame
                and odom_frame != self._latest_odom_frame):
            self.get_logger().warn(
                f'[ctrl/carrot] FRAME MISMATCH: odom in {odom_frame!r}, '
                f'carrot lifted using {self._latest_odom_frame!r}',
                throttle_duration_sec=5.0)

        now = self.get_clock().now()
        age_s = (now - self._carrot_last_seen).nanoseconds * 1e-9
        marker_lost = age_s > self.marker_timeout

        if self._prev_cmd_time is None:
            dt = 0.0
        else:
            dt = (now - self._prev_cmd_time).nanoseconds * 1e-9
            if dt < 0.0:
                dt = 0.0
        self._prev_cmd_time = now

        ryaw = yaw_from_quaternion(robot_pose.orientation)
        bearing = math.atan2(
            target.position.y - robot_pose.position.y,
            target.position.x - robot_pose.position.x,
        )
        heading_error = math.atan2(
            math.sin(bearing - ryaw), math.cos(bearing - ryaw))

        target_w = max(
            -self.max_angular_velocity,
            min(self.max_angular_velocity, self.angular_gain * heading_error),
        )

        if not marker_lost:
            # Marker visible: stop at the standoff (never closer, never back).
            gap = distance_to_target - self.desired_carrot_distance
            if gap > 0.0:
                target_v = self._carrot_target_velocity(gap, heading_error)
                phase = 'FOLLOW'
            else:
                target_v = 0.0
                phase = 'HOLD'
        else:
            # Marker lost: drive to the last known position, then rotate to
            # the last known marker yaw, then stop.
            if distance_to_target > self.goal_tolerance:
                target_v = self._carrot_target_velocity(
                    distance_to_target, heading_error)
                phase = 'LOST_APPROACH'
            else:
                target_v = 0.0
                yaw_error = math.atan2(
                    math.sin(self._carrot_yaw - ryaw),
                    math.cos(self._carrot_yaw - ryaw))
                if abs(yaw_error) > self.final_yaw_tolerance:
                    target_w = max(
                        -self.max_angular_velocity,
                        min(self.max_angular_velocity,
                            self.angular_gain * yaw_error),
                    )
                    phase = 'LOST_ALIGN'
                else:
                    target_w = 0.0
                    phase = 'LOST_DONE'

        # Apply linear slew limit so accel/decel never exceed the configured
        # rates, regardless of phase transitions.
        v_cmd = self._slew_linear(target_v, dt)
        self._prev_linear_cmd = v_cmd

        cmd = Twist()
        cmd.linear.x = v_cmd
        cmd.angular.z = target_w

        rx = robot_pose.position.x
        ry = robot_pose.position.y
        tx = target.position.x
        ty = target.position.y
        self.get_logger().info(
            f'[ctrl/carrot:{phase}] PUB cmd_vel '
            f'lin={cmd.linear.x:+.3f} (tgt={target_v:+.3f}) '
            f'ang={cmd.angular.z:+.3f} '
            f'| robot=({rx:.2f},{ry:.2f}) yaw={math.degrees(ryaw):+.1f}deg '
            f'| target=({tx:.2f},{ty:.2f}) '
            f'bearing={math.degrees(bearing):+.1f}deg '
            f'heading_err={math.degrees(heading_error):+.1f}deg '
            f'| dist={distance_to_target:.2f} '
            f'standoff={self.desired_carrot_distance:.2f} '
            f'marker_age={age_s:.2f}s lost={marker_lost} '
            f'subs={self.cmd_vel_pub.get_subscription_count()}')

        self._publish_cmd(cmd)

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
        # TODO: Sim vs Helhest sign
        heading_error = math.atan2(
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

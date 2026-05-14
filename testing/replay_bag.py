#!/usr/bin/env python3
"""Drive the waypoint_follower from a rosbag by mimicking whycode.

Standalone test helper — NOT part of the package build. Run it manually
after sourcing the workspace:

    source install/setup.bash
    python3 testing/replay_bag.py /path/to/bag --odom-in-topic /recorded_odometry

Reads a single Odometry topic from the bag and publishes each pose as a
single-marker ``whycode_interfaces/MarkerArray`` on the same topic the real
whycode_node uses. The marker pose is expressed in the live base_link frame
(via the inverse of the most recent live odometry) so that the navigator's
camera→odom transform reconstructs the original bagged pose.

Requires the simulator's odometry source to be running on the same topic
the navigator subscribes to (``--live-odom-topic``).
"""

import argparse
import math
import sys
import time
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message
from whycode_interfaces.msg import Marker, MarkerArray


def _best_effort(depth: int) -> QoSProfile:
    return QoSProfile(
        depth=depth,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
    )


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _pose_in_base(pose_in_odom: Pose, robot_in_odom: Pose) -> Pose:
    """Express an odom-frame pose in the robot's current base_link frame
    (inverse of the planar rigid transform the navigator applies)."""
    yaw = _yaw_from_quaternion(robot_in_odom.orientation)
    c, s = math.cos(yaw), math.sin(yaw)
    dx = pose_in_odom.position.x - robot_in_odom.position.x
    dy = pose_in_odom.position.y - robot_in_odom.position.y
    out = Pose()
    out.position.x = c * dx + s * dy
    out.position.y = -s * dx + c * dy
    out.position.z = pose_in_odom.position.z - robot_in_odom.position.z
    rel_yaw = _yaw_from_quaternion(pose_in_odom.orientation) - yaw
    out.orientation.z = math.sin(rel_yaw / 2.0)
    out.orientation.w = math.cos(rel_yaw / 2.0)
    return out


def _read_odometry(bag_path: str, topic: str) -> List[Tuple[int, Odometry]]:
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_path, storage_id=''),
        ConverterOptions(input_serialization_format='cdr',
                         output_serialization_format='cdr'),
    )
    type_by_topic = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in type_by_topic:
        raise RuntimeError(
            f"Topic '{topic}' not in bag. Available: {list(type_by_topic)}")
    if type_by_topic[topic] != 'nav_msgs/msg/Odometry':
        raise RuntimeError(
            f"Topic '{topic}' is {type_by_topic[topic]}, expected nav_msgs/msg/Odometry")

    msg_cls = get_message(type_by_topic[topic])
    out: List[Tuple[int, Odometry]] = []
    while reader.has_next():
        t, data, stamp_ns = reader.read_next()
        if t == topic:
            out.append((stamp_ns, deserialize_message(data, msg_cls)))
    return out


class BagReplay(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__('rosbag_replay')
        self.args = args
        self._latest_odom: Optional[Pose] = None
        self.create_subscription(
            Odometry, args.live_odom_topic, self._odom_cb, _best_effort(10))
        self.markers_pub = self.create_publisher(
            MarkerArray, args.marker_topic, _best_effort(200))

    def _odom_cb(self, msg: Odometry) -> None:
        self._latest_odom = msg.pose.pose

    def _build_marker_array(self, bagged: Odometry) -> MarkerArray:
        rel = _pose_in_base(bagged.pose.pose, self._latest_odom)
        marker = Marker()
        marker.position = rel
        out = MarkerArray()
        out.header.stamp = bagged.header.stamp
        out.header.frame_id = 'base_link'
        out.markers.append(marker)
        return out

    def run(self) -> int:
        self.get_logger().info(f'Reading {self.args.bag} ...')
        messages = _read_odometry(self.args.bag, self.args.odom_in_topic)
        if not messages:
            self.get_logger().error(
                f'No messages on {self.args.odom_in_topic} in bag.')
            return 1
        self.get_logger().info(f'Loaded {len(messages)} odometry messages.')

        # Wait for both: a subscriber on the marker topic AND a first live
        # odometry snapshot — we need the latter to compute base_link poses.
        deadline = time.monotonic() + self.args.subscriber_timeout
        while time.monotonic() < deadline and (
                self.markers_pub.get_subscription_count() == 0
                or self._latest_odom is None):
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.markers_pub.get_subscription_count() == 0:
            self.get_logger().warn(
                f'No subscriber on {self.args.marker_topic}; publishing anyway.')
        if self._latest_odom is None:
            self.get_logger().error(
                f'No live odometry on {self.args.live_odom_topic} after '
                f'{self.args.subscriber_timeout}s; aborting.')
            return 2

        sent = 0
        for _, bagged in messages:
            if self._latest_odom is None:  # defensive — should not regress
                rclpy.spin_once(self, timeout_sec=0.02)
                continue
            self.markers_pub.publish(self._build_marker_array(bagged))
            sent += 1
            rclpy.spin_once(self, timeout_sec=0.02)
        self.get_logger().info(
            f'Sent {sent} marker arrays to {self.args.marker_topic}.')
        time.sleep(0.5)  # grace period before shutdown
        return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', help='Path to a rosbag2 directory.')
    parser.add_argument('--odom-in-topic', default='/odom',
                        help='Odometry topic name inside the bag (used to derive markers).')
    parser.add_argument('--live-odom-topic', default='/odometry_publisher',
                        help='Live odometry topic the navigator subscribes to '
                             '(used to express each bagged pose in base_link).')
    parser.add_argument('--marker-topic', default='/whycode_node/markers',
                        help='Topic to publish synthesized MarkerArrays on.')
    parser.add_argument('--subscriber-timeout', type=float, default=5.0,
                        help='Seconds to wait for a marker subscriber and live odom.')
    args = parser.parse_args(argv)

    rclpy.init()
    node = BagReplay(args)
    try:
        return node.run()
    except KeyboardInterrupt:
        return 130
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Seed the waypoint_follower with waypoints from a rosbag.

Standalone test helper — NOT part of the package build. Run it manually
after sourcing the workspace:

    source install/setup.bash
    python3 testing/replay_bag.py /path/to/bag --odom-in-topic /recorded_odometry

It reads a single Odometry topic from a rosbag2 directory and publishes
each pose as a PoseStamped waypoint (preserving the original header
stamps so the follower can derive segment velocity from them). The
follower's buffer down-samples by its own ``waypoint_spacing`` parameter,
so this script publishes everything verbatim.

It does NOT publish odometry — the simulator is expected to provide the
robot's pose feedback to the follower. Once the waypoints are sent the
script exits; the follower keeps them in its buffer and drives toward
them using sim odometry as feedback.
"""

import argparse
import sys
import time
from typing import List, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


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


def _to_waypoint(odom: Odometry) -> PoseStamped:
    wp = PoseStamped()
    wp.header = odom.header
    wp.pose = odom.pose.pose
    return wp


class BagReplay(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__('rosbag_replay')
        self.args = args
        wp_qos = QoSProfile(
            depth=200,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.wp_pub = self.create_publisher(
            PoseStamped, args.waypoint_topic, wp_qos)

    def run(self) -> int:
        self.get_logger().info(f'Reading {self.args.bag} ...')
        messages = _read_odometry(self.args.bag, self.args.odom_in_topic)
        if not messages:
            self.get_logger().error(
                f'No messages on {self.args.odom_in_topic} in bag.')
            return 1
        self.get_logger().info(f'Loaded {len(messages)} odometry messages.')

        waypoints = [_to_waypoint(odom) for _, odom in messages]

        # Wait for the follower to subscribe before sending waypoints.
        deadline = time.monotonic() + self.args.subscriber_timeout
        while (self.wp_pub.get_subscription_count() == 0
               and time.monotonic() < deadline):
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.wp_pub.get_subscription_count() == 0:
            self.get_logger().warn(
                f'No subscriber on {self.args.waypoint_topic}; publishing anyway.')

        for wp in waypoints:
            self.wp_pub.publish(wp)
            time.sleep(0.02)
        self.get_logger().info(
            f'Sent {len(waypoints)} waypoints to {self.args.waypoint_topic}.')
        # Brief grace period so the last messages drain before shutdown.
        time.sleep(0.5)
        return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', help='Path to a rosbag2 directory.')
    parser.add_argument('--odom-in-topic', default='/odom',
                        help='Odometry topic name inside the bag (used to derive waypoints).')
    parser.add_argument('--waypoint-topic', default='waypoint',
                        help='Topic to publish derived waypoints on.')
    parser.add_argument('--subscriber-timeout', type=float, default=5.0,
                        help='Seconds to wait for a waypoint subscriber.')
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

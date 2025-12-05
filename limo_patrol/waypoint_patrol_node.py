#!/usr/bin/env python3
import math
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool
from nav2_msgs.action import FollowWaypoints

import yaml


class WaypointPatrolNode(Node):
    """
    - config/waypoints.yaml에서 다중 waypoint를 읽어
      Nav2 FollowWaypoints 액션으로 순찰.
    - /fire_detected, /patrol/obstacle_detected 구독:
      * fire_detected=True  -> Nav2 goal cancel + cmd_vel=0 정지
      * obstacle_detected=True -> cmd_vel=0 (안전 정지)
    """

    def __init__(self):
        super().__init__('waypoint_patrol_node')

        self.declare_parameter(
            'waypoint_file',
            str(Path.home() / 'limo_patrol_waypoints.yaml')
        )
        waypoint_file = self.get_parameter('waypoint_file').get_parameter_value().string_value

        self.waypoints = self.load_waypoints(waypoint_file)
        if not self.waypoints:
            self.get_logger().error('No waypoints loaded. Check waypoint_file.')
        else:
            self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints.')

        # 상태 플래그
        self.fire_detected = False
        self.obstacle_detected = False

        # 화재 / 장애물 구독
        self.create_subscription(Bool, 'fire_detected', self.fire_callback, 10)
        self.create_subscription(Bool, '/patrol/obstacle_detected', self.obstacle_callback, 10)

        # cmd_vel 퍼블리셔
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # FollowWaypoints 액션 클라이언트
        self._client = ActionClient(self, FollowWaypoints, 'follow_waypoints')
        self._goal_handle = None

        # Nav2 서버 준비되면 goal 한 번 전송
        self.timer = self.create_timer(2.0, self.try_send_goal_once)
        self.sent = False

    # ----------------------- 콜백 -----------------------
    def fire_callback(self, msg: Bool):
        if msg.data and not self.fire_detected:
            self.get_logger().warn('🔥 Fire detected! Stopping patrol and canceling Nav2 goal.')
            self.fire_detected = True
            self.stop_robot()
            self.cancel_nav_goal()
            # TODO: 텔레그램 알림 연동
        elif not msg.data and self.fire_detected:
            self.get_logger().info('Fire cleared.')
            self.fire_detected = False

    def obstacle_callback(self, msg: Bool):
        if msg.data and not self.obstacle_detected:
            self.get_logger().warn('Obstacle detected. Stopping robot (safety brake).')
            self.obstacle_detected = True
            self.stop_robot()
        elif not msg.data and self.obstacle_detected:
            self.get_logger().info('Obstacle cleared.')
            self.obstacle_detected = False

    # -------------------- 유틸 함수 ---------------------
    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    def cancel_nav_goal(self):
        if self._goal_handle is None:
            self.get_logger().warn('No active FollowWaypoints goal to cancel.')
            return
        future = self._goal_handle.cancel_goal_async()
        future.add_done_callback(self._cancel_done_callback)

    def _cancel_done_callback(self, future):
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info('FollowWaypoints goal successfully canceled.')
        else:
            self.get_logger().warn('FollowWaypoints goal cancel not accepted.')

    # ----------------- Waypoint 로드 --------------------
    def load_waypoints(self, path):
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f'Failed to load waypoint file: {e}')
            return []

        waypoints = []
        for wp in data.get('waypoints', []):
            pose_data = wp['pose']
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(pose_data['x'])
            pose.pose.position.y = float(pose_data['y'])
            yaw = float(pose_data['yaw'])
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            waypoints.append(pose)
        return waypoints

    # ----------------- Nav2 Goal 전송 -------------------
    def try_send_goal_once(self):
        if self.sent:
            return
        if not self._client.wait_for_server(timeout_sec=0.5):
            self.get_logger().info('Waiting for FollowWaypoints action server...')
            return

        if not self.waypoints:
            self.get_logger().error('No waypoints to send.')
            self.timer.cancel()
            return

        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = self.waypoints

        self.get_logger().info(f'Sending {len(self.waypoints)} waypoints to Nav2.')

        send_future = self._client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.goal_response_callback)
        self.sent = True
        self.timer.cancel()

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('FollowWaypoints goal rejected.')
            return

        self.get_logger().info('FollowWaypoints goal accepted.')
        self._goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(
            f'FollowWaypoints finished. '
            f'failed_waypoints={result.failed_waypoints}, failed_ids={result.failed_ids}'
        )
        self._goal_handle = None


def main(args=None):
    rclpy.init(args=args)
    node = WaypointPatrolNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('WaypointPatrolNode shutting down...')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

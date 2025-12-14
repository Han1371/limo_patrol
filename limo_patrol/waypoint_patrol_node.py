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
    - Nav2 FollowWaypoints로 waypoint 순찰
    - /patrol/pause True: goal cancel + 정지
    - fire_detected True: goal cancel + 정지
    - /patrol/emergency_stop True: goal cancel + 정지 + latch(재개 불가)
    - waypoint 1바퀴 완료 시 /patrol/cycle_done=True 발행 후 다음 바퀴 재시작
    """

    def __init__(self):
        super().__init__('waypoint_patrol_node')

        self.declare_parameter('waypoint_file', str(Path.home() / 'limo_patrol_waypoints.yaml'))
        self.declare_parameter('loop_patrol', True)
        self.declare_parameter('restart_delay_sec', 1.0)

        waypoint_file = self.get_parameter('waypoint_file').value
        self.loop_patrol = bool(self.get_parameter('loop_patrol').value)
        self.restart_delay_sec = float(self.get_parameter('restart_delay_sec').value)

        self.waypoints = self.load_waypoints(waypoint_file)
        if not self.waypoints:
            self.get_logger().error('No waypoints loaded.')
        else:
            self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints.')

        self.fire_detected = False
        self.paused = False
        self.emergency_stop = False

        self._goal_handle = None
        self.sent = False

        # publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cycle_done_pub = self.create_publisher(Bool, '/patrol/cycle_done', 10)

        # subscribers
        self.create_subscription(Bool, 'fire_detected', self.fire_callback, 10)
        self.create_subscription(Bool, '/patrol/pause', self.pause_callback, 10)
        self.create_subscription(Bool, '/patrol/emergency_stop', self.emergency_stop_callback, 10)

        # action
        self._client = ActionClient(self, FollowWaypoints, 'follow_waypoints')

        # restart timer(중복 방지)
        self.restart_timer = None

        # initial attempt
        self.init_timer = self.create_timer(2.0, self.try_send_goal_once)

    # ---------------- callbacks ----------------
    def emergency_stop_callback(self, msg: Bool):
        if msg.data and not self.emergency_stop:
            self.emergency_stop = True
            self.get_logger().error('🛑 EMERGENCY STOP! Canceling goal and stopping permanently.')
            self.stop_robot()
            self.cancel_nav_goal()
            self.sent = True
        elif (not msg.data) and self.emergency_stop:
            self.get_logger().warn('Emergency stop is latched. Ignore stop=False.')

    def fire_callback(self, msg: Bool):
        if msg.data and not self.fire_detected:
            self.fire_detected = True
            self.get_logger().warn('🔥 Fire detected! Canceling goal and stopping.')
            self.stop_robot()
            self.cancel_nav_goal()
        elif (not msg.data) and self.fire_detected:
            self.fire_detected = False
            self.get_logger().info('Fire cleared.')
            self.maybe_resume('fire_cleared')

    def pause_callback(self, msg: Bool):
        if msg.data and not self.paused:
            self.paused = True
            self.get_logger().warn('Patrol PAUSE received. Canceling goal and stopping.')
            self.stop_robot()
            self.cancel_nav_goal()
        elif (not msg.data) and self.paused:
            self.paused = False
            self.get_logger().info('Patrol RESUME received.')
            self.maybe_resume('pause_cleared')

    # ---------------- helpers ----------------
    def can_move(self) -> bool:
        return (not self.emergency_stop) and (not self.fire_detected) and (not self.paused)

    def stop_robot(self):
        t = Twist()
        t.linear.x = 0.0
        t.angular.z = 0.0
        self.cmd_pub.publish(t)

    def cancel_nav_goal(self):
        if self._goal_handle is None:
            return
        try:
            self._goal_handle.cancel_goal_async()
        except Exception:
            pass

    def schedule_restart(self):
        if self.restart_timer is not None:
            try:
                self.restart_timer.cancel()
            except Exception:
                pass
            self.restart_timer = None
        self.restart_timer = self.create_timer(self.restart_delay_sec, self._restart_once)

    def _restart_once(self):
        if self.restart_timer is not None:
            try:
                self.restart_timer.cancel()
            except Exception:
                pass
            self.restart_timer = None
        self.try_send_goal_once()

    def maybe_resume(self, reason: str):
        if not self.loop_patrol:
            return
        if not self.can_move():
            return
        if self._goal_handle is not None:
            return
        self.get_logger().info(f'Restarting patrol due to {reason}')
        self.sent = False
        self.schedule_restart()

    # ---------------- waypoint load ----------------
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

    # ---------------- goal send ----------------
    def try_send_goal_once(self):
        if self.sent:
            return
        if not self.can_move():
            return
        if not self._client.wait_for_server(timeout_sec=0.5):
            self.get_logger().info('Waiting for FollowWaypoints action server...')
            return
        if not self.waypoints:
            self.get_logger().error('No waypoints to send.')
            return

        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = self.waypoints

        self.get_logger().info(f'Sending {len(self.waypoints)} waypoints to Nav2.')
        fut = self._client.send_goal_async(goal_msg)
        fut.add_done_callback(self.goal_response_callback)

        self.sent = True

    def goal_response_callback(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error('Goal rejected.')
            self.sent = False
            self.schedule_restart()
            return

        self.get_logger().info('Goal accepted.')
        self._goal_handle = gh
        res_fut = gh.get_result_async()
        res_fut.add_done_callback(self.result_callback)

    def result_callback(self, future):
        res = future.result().result
        self.get_logger().info(
            f'FollowWaypoints finished. failed_waypoints={res.failed_waypoints}, failed_ids={res.failed_ids}'
        )
        self._goal_handle = None

        # cycle_done trigger
        self.cycle_done_pub.publish(Bool(data=True))
        self.get_logger().info('Published /patrol/cycle_done=True')

        # loop
        if self.loop_patrol and self.can_move():
            self.get_logger().info('Loop patrol -> restarting.')
            self.sent = False
            self.schedule_restart()


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

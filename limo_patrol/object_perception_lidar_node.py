#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String

import requests


class ObjectPerceptionLidarNode(Node):
    """
    LiDAR 기반 전방 장애물 감지:
      - 전방 +/- front_angle_deg 영역에서 obstacle_range_m 이내 비율(min_hit_ratio) 이상이면 장애물
      - 장애물 상태 동안 cmd_vel=0을 주기적으로 publish하여 Nav2 cmd_vel을 덮어씀
      - 장애물 감지 시 텔레그램 경고 1회
    """

    def __init__(self):
        super().__init__('object_perception_lidar_node')

        # ---- Parameters ----
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        self.declare_parameter('front_angle_deg', 30.0)     # 전방 +/- 각도
        self.declare_parameter('min_range_m', 0.12)         # 노이즈 제거(너무 가까운 값 무시)
        self.declare_parameter('obstacle_range_m', 0.55)    # 장애물 판단 거리
        self.declare_parameter('min_hit_ratio', 0.08)       # 전방 유효빔 중 이 비율 이상이 가까우면 장애물

        self.declare_parameter('stop_publish_hz', 10.0)

        # Telegram
        self.declare_parameter('telegram_enable', True)
        self.declare_parameter('telegram_bot_token', '')
        self.declare_parameter('telegram_chat_id', '')
        self.declare_parameter('telegram_prefix', '[LIMO]')

        self.scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value

        self.front_angle_deg = float(self.get_parameter('front_angle_deg').value)
        self.min_range_m = float(self.get_parameter('min_range_m').value)
        self.obstacle_range_m = float(self.get_parameter('obstacle_range_m').value)
        self.min_hit_ratio = float(self.get_parameter('min_hit_ratio').value)
        self.stop_publish_hz = float(self.get_parameter('stop_publish_hz').value)

        self.telegram_enable = bool(self.get_parameter('telegram_enable').value)
        self.bot_token = self.get_parameter('telegram_bot_token').get_parameter_value().string_value
        self.chat_id = self.get_parameter('telegram_chat_id').get_parameter_value().string_value
        self.tg_prefix = self.get_parameter('telegram_prefix').get_parameter_value().string_value

        # ---- Publishers/Subscribers ----
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.obstacle_pub = self.create_publisher(Bool, '/patrol/obstacle_detected', 10)
        self.event_pub = self.create_publisher(String, '/patrol/event', 10)

        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, 10)

        # ---- State ----
        self._obstacle = False
        self._alert_sent = False

        # stop timer
        period = 1.0 / max(self.stop_publish_hz, 1.0)
        self.stop_timer = self.create_timer(period, self.on_stop_timer)

        self.get_logger().info(
            f"[ObjectPerceptionLidarNode]\n"
            f"- scan_topic: {self.scan_topic}\n"
            f"- cmd_vel_topic: {self.cmd_vel_topic}\n"
            f"- obstacle_range_m: {self.obstacle_range_m}\n"
            f"- front_angle_deg: +/-{self.front_angle_deg}\n"
            f"- min_hit_ratio: {self.min_hit_ratio}\n"
        )

    def on_scan(self, scan: LaserScan):
        # 전방 각도 범위를 인덱스로 변환
        front = math.radians(self.front_angle_deg)
        a_min = scan.angle_min
        inc = scan.angle_increment

        # [-front, +front]
        i0 = int(round(((-front) - a_min) / inc))
        i1 = int(round(((+front) - a_min) / inc))

        i0 = max(0, min(i0, len(scan.ranges) - 1))
        i1 = max(0, min(i1, len(scan.ranges) - 1))
        if i0 > i1:
            i0, i1 = i1, i0

        window = scan.ranges[i0:i1 + 1]
        if not window:
            return

        valid = []
        for r in window:
            if math.isfinite(r) and (self.min_range_m < r < scan.range_max):
                valid.append(r)

        if not valid:
            # 유효 값이 없으면 장애물 아님 처리
            self.set_obstacle(False)
            return

        hits = sum(1 for r in valid if r < self.obstacle_range_m)
        ratio = hits / max(len(valid), 1)

        detected = (ratio >= self.min_hit_ratio)
        self.set_obstacle(detected)

    def set_obstacle(self, detected: bool):
        if detected == self._obstacle:
            return

        self._obstacle = detected

        b = Bool()
        b.data = detected
        self.obstacle_pub.publish(b)

        ev = String()
        if detected:
            ev.data = "OBSTACLE_DETECTED"
            self.event_pub.publish(ev)

            if not self._alert_sent:
                self._alert_sent = True
                self.send_telegram(f"{self.tg_prefix} LiDAR 장애물 감지! 로봇 정지 및 경고 전송. (순찰 일시중지)")
        else:
            ev.data = "OBSTACLE_CLEARED"
            self.event_pub.publish(ev)

            # 다음 감지 대비
            self._alert_sent = False
            self.send_telegram(f"{self.tg_prefix} 장애물 해제! 순찰 재개 가능.")

        self.get_logger().info(f"Obstacle changed -> {detected}")

    def on_stop_timer(self):
        if not self._obstacle:
            return

        # 장애물 상태에서는 0속도 계속 publish
        t = Twist()
        t.linear.x = 0.0
        t.linear.y = 0.0
        t.linear.z = 0.0
        t.angular.x = 0.0
        t.angular.y = 0.0
        t.angular.z = 0.0
        self.cmd_pub.publish(t)

    def send_telegram(self, text: str):
        if not self.telegram_enable:
            return
        if not self.bot_token or not self.chat_id:
            self.get_logger().warn("telegram_enable=True 이지만 telegram_bot_token/telegram_chat_id가 비어있음.")
            return

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": text}
            resp = requests.post(url, json=payload, timeout=2.5)
            if resp.status_code != 200:
                self.get_logger().warn(f"Telegram send failed: {resp.status_code}, {resp.text}")
        except Exception as e:
            self.get_logger().warn(f"Telegram exception: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectPerceptionLidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

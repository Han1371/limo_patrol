#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np


class ModeManager(Node):
    def __init__(self):
        super().__init__('mode_manager_node')
        self.bridge = CvBridge()

        # mode_source: manual / auto
        self.declare_parameter('mode_source', 'manual')
        self.declare_parameter('initial_mode', 'NIGHT')
        self.declare_parameter('brightness_topic', '/camera/image_raw')
        self.declare_parameter('brightness_threshold', 50.0)

        # ✅ 핵심: 1회 결정/발행
        self.declare_parameter('publish_once', True)
        self.declare_parameter('wait_first_image_timeout_sec', 5.0)

        self.mode_source = self.get_parameter('mode_source').get_parameter_value().string_value
        self.mode = self.get_parameter('initial_mode').get_parameter_value().string_value
        brightness_topic = self.get_parameter('brightness_topic').get_parameter_value().string_value
        self.night_threshold = self.get_parameter('brightness_threshold').get_parameter_value().double_value

        self.publish_once = bool(self.get_parameter('publish_once').value)
        self.wait_first_image_timeout_sec = float(self.get_parameter('wait_first_image_timeout_sec').value)

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.pub = self.create_publisher(String, '/patrol_mode', qos)

        self._published = False
        self._start_wall = time.time()

        if self.mode_source == 'auto':
            self.sub_img = self.create_subscription(Image, brightness_topic, self.image_cb, 10)
            self.get_logger().info(
                f"ModeManager(auto) started, night_threshold={self.night_threshold}, publish_once={self.publish_once}"
            )
        else:
            self.get_logger().info(f"ModeManager(manual) started, mode={self.mode}, publish_once={self.publish_once}")

        self.timer = self.create_timer(0.2, self.timer_cb)

    def image_cb(self, msg: Image):
        if self.publish_once and self._published:
            return

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))

        prev_mode = self.mode
        self.mode = 'NIGHT' if mean_brightness < self.night_threshold else 'DAY'

        if self.mode != prev_mode:
            self.get_logger().info(
                f"Mode changed: {prev_mode} -> {self.mode} (brightness={mean_brightness:.1f})"
            )

        # auto + publish_once => 첫 이미지 기반 확정 후 1회 발행하고 끝
        if self.publish_once and not self._published:
            self.publish_once_and_finish()

    def timer_cb(self):
        if self._published and self.publish_once:
            return

        # manual이면 즉시 1회 발행 가능
        if self.mode_source != 'auto':
            if self.publish_once:
                self.publish_once_and_finish()
            else:
                self.publish_periodic()
            return

        # auto + publish_once => 이미지 안 오면 timeout 후 initial_mode로 1회 발행
        if self.publish_once:
            if (time.time() - self._start_wall) > self.wait_first_image_timeout_sec:
                self.get_logger().warn(
                    f"No image within {self.wait_first_image_timeout_sec}s. Fallback to initial_mode={self.mode}"
                )
                self.publish_once_and_finish()
            return

        # auto + periodic
        self.publish_periodic()

    def publish_periodic(self):
        msg = String()
        msg.data = self.mode
        self.pub.publish(msg)

    def publish_once_and_finish(self):
        msg = String()
        msg.data = self.mode
        self.pub.publish(msg)
        self._published = True

        self.get_logger().info(f"Mode fixed and published once: {self.mode}")

        # 구독/타이머 정리(모드 고정)
        try:
            if hasattr(self, 'sub_img'):
                self.destroy_subscription(self.sub_img)
        except Exception:
            pass
        try:
            self.destroy_timer(self.timer)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = ModeManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()

# limo_night_patrol/mode_manager_node.py

import rclpy
from rclpy.node import Node
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
        self.declare_parameter('brightness_threshold', 50.0)  # 평균 밝기 기준

        self.mode_source = self.get_parameter('mode_source').get_parameter_value().string_value
        self.mode = self.get_parameter('initial_mode').get_parameter_value().string_value
        brightness_topic = self.get_parameter('brightness_topic').get_parameter_value().string_value
        self.night_threshold = self.get_parameter('brightness_threshold').get_parameter_value().double_value

        self.pub = self.create_publisher(String, '/patrol_mode', 10)
        self.timer = self.create_timer(1.0, self.timer_cb)

        if self.mode_source == 'auto':
            self.sub_img = self.create_subscription(Image, brightness_topic, self.image_cb, 10)
            self.get_logger().info(
                f"ModeManager(auto) started, night_threshold={self.night_threshold}"
            )
        else:
            self.get_logger().info(f"ModeManager(manual) started, mode={self.mode}")

    def image_cb(self, msg: Image):
        # 이미지 평균 밝기 계산
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))

        prev_mode = self.mode
        if mean_brightness < self.night_threshold:
            self.mode = 'NIGHT'
        else:
            self.mode = 'DAY'

        if self.mode != prev_mode:
            self.get_logger().info(
                f"Mode changed: {prev_mode} -> {self.mode} (brightness={mean_brightness:.1f})"
            )

    def timer_cb(self):
        msg = String()
        msg.data = self.mode
        self.pub.publish(msg)


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

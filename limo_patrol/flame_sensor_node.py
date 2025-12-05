#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from std_msgs.msg import Int32, Bool


class FlameSensorNode(Node):
    """
    아두이노 불꽃 감지 센서 값을 구독해서
    임계값(threshold) 기준으로 화재 감지 여부를 판단하는 노드.
    - 입력 : /flame_raw (std_msgs/Int32)
    - 출력 : /fire_detected (std_msgs/Bool)
    """

    def __init__(self):
        super().__init__('flame_sensor_node')

        self.declare_parameter('raw_topic', '/flame_raw')
        self.declare_parameter('flame_threshold', 500)

        self.raw_topic = self.get_parameter('raw_topic').get_parameter_value().string_value
        self.flame_threshold = self.get_parameter('flame_threshold').get_parameter_value().integer_value

        self.raw_sub = self.create_subscription(
            Int32,
            self.raw_topic,
            self.raw_callback,
            10
        )

        self.fire_pub = self.create_publisher(Bool, 'fire_detected', 10)

        self.last_fire_state = False

        self.get_logger().info(
            f'FlameSensorNode started. Subscribing [{self.raw_topic}], '
            f'threshold={self.flame_threshold}'
        )

    def raw_callback(self, msg: Int32):
        raw_value = msg.data
        fire_detected = raw_value >= self.flame_threshold

        fire_msg = Bool()
        fire_msg.data = fire_detected
        self.fire_pub.publish(fire_msg)

        if fire_detected != self.last_fire_state:
            self.last_fire_state = fire_detected
            if fire_detected:
                self.get_logger().warn(
                    f'🔥 Flame detected! raw={raw_value}, threshold={self.flame_threshold}'
                )
            else:
                self.get_logger().info(
                    f'Flame cleared. raw={raw_value}, threshold={self.flame_threshold}'
                )


def main(args=None):
    rclpy.init(args=args)
    node = FlameSensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('FlameSensorNode shutting down...')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

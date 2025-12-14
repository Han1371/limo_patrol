#!/usr/bin/env python3
import requests

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Bool


class FlameSensorNode(Node):
    """
    - 입력 : /flame_raw (std_msgs/Int32)
    - 출력 : /fire_detected (std_msgs/Bool)
    - 화재 감지 상태 변화 시 텔레그램 메시지 전송
    """

    def __init__(self):
        super().__init__('flame_sensor_node')

        self.declare_parameter('raw_topic', '/flame_raw')
        self.declare_parameter('flame_threshold', 500)

        # Telegram
        self.declare_parameter('telegram_bot_token', '')
        self.declare_parameter('telegram_chat_id', '')
        self.declare_parameter('send_clear_message', True)

        self.raw_topic = self.get_parameter('raw_topic').value
        self.flame_threshold = int(self.get_parameter('flame_threshold').value)

        self.bot_token = self.get_parameter('telegram_bot_token').value
        self.chat_id = self.get_parameter('telegram_chat_id').value
        self.send_clear = bool(self.get_parameter('send_clear_message').value)

        self.create_subscription(Int32, self.raw_topic, self.raw_callback, 10)
        self.fire_pub = self.create_publisher(Bool, 'fire_detected', 10)

        self.last_fire_state = False

        self.get_logger().info(
            f'FlameSensorNode started. Subscribing [{self.raw_topic}], threshold={self.flame_threshold}'
        )

    def raw_callback(self, msg: Int32):
        raw_value = msg.data
        fire_detected = raw_value >= self.flame_threshold

        # 항상 발행 (다른 노드가 상태를 계속 알 수 있게)
        self.fire_pub.publish(Bool(data=fire_detected))

        # 상태 변화 시에만 알림
        if fire_detected == self.last_fire_state:
            return

        self.last_fire_state = fire_detected
        if fire_detected:
            self.get_logger().warn(f'🔥 Flame detected! raw={raw_value}, threshold={self.flame_threshold}')
            self.send_telegram_text(
                f'🔥 [LIMO] 화재 감지!\nraw={raw_value} / threshold={self.flame_threshold}'
            )
        else:
            self.get_logger().info(f'Flame cleared. raw={raw_value}')
            if self.send_clear:
                self.send_telegram_text(f'✅ [LIMO] 화재 해제.\nraw={raw_value}')

    def send_telegram_text(self, text: str) -> bool:
        if not self.bot_token or not self.chat_id:
            self.get_logger().error('Telegram token/chat_id is empty. Set parameters.')
            return False

        url = f'https://api.telegram.org/bot{self.bot_token}/sendMessage'
        try:
            resp = requests.post(url, data={'chat_id': self.chat_id, 'text': text}, timeout=10)
            if resp.status_code == 200:
                return True
            self.get_logger().error(f'Telegram HTTP {resp.status_code}: {resp.text[:200]}')
            return False
        except Exception as e:
            self.get_logger().error(f'Telegram send exception: {e}')
            return False


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

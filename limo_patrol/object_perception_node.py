#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, String
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from cv_bridge import CvBridge


class ObjectPerceptionNode(Node):
    """
    조도 파라미터에 따라 day/night 모드를 결정하고,
    - day  : YOLO detection을 이용해 객체 인식
    - night: depth 카메라를 이용해 ROI 최소 거리 기반 장애물 인식
    결과를 /patrol/obstacle_detected, /patrol/mode 로 퍼블리시.
    """

    def __init__(self):
        super().__init__('object_perception_node')

        # 파라미터
        self.declare_parameter('illumination', 50.0)
        self.declare_parameter('illumination_threshold', 100.0)
        self.declare_parameter('yolo_topic', '/yolov5/detections')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('max_detection_distance', 2.0)

        self.illumination = float(self.get_parameter('illumination').value)
        self.illumination_threshold = float(self.get_parameter('illumination_threshold').value)
        self.yolo_topic = self.get_parameter('yolo_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.max_detection_distance = float(self.get_parameter('max_detection_distance').value)

        # 모드 결정
        if self.illumination < self.illumination_threshold:
            self.mode = 'day'
        else:
            self.mode = 'night'

        # 모드 퍼블리셔 (다른 노드 참고용)
        self.mode_pub = self.create_publisher(String, '/patrol/mode', 10)
        self.publish_mode()

        self.get_logger().info(
            f'ObjectPerceptionNode started. illumination={self.illumination}, '
            f'threshold={self.illumination_threshold} -> mode={self.mode}'
        )

        # 장애물 퍼블리셔
        self.obstacle_pub = self.create_publisher(Bool, '/patrol/obstacle_detected', 10)

        self.bridge = CvBridge()
        self.yolo_sub = None
        self.depth_sub = None

        if self.mode == 'day':
            self.yolo_sub = self.create_subscription(
                Detection2DArray,
                self.yolo_topic,
                self.yolo_callback,
                10
            )
            self.get_logger().info(f'DAY mode: Subscribing YOLO [{self.yolo_topic}]')
        else:
            self.depth_sub = self.create_subscription(
                Image,
                self.depth_topic,
                self.depth_callback,
                10
            )
            self.get_logger().info(
                f'NIGHT mode: Subscribing depth [{self.depth_topic}], '
                f'max_detection_distance={self.max_detection_distance} m'
            )

    def publish_mode(self):
        msg = String()
        msg.data = self.mode
        self.mode_pub.publish(msg)

    # --------------------- DAY MODE: YOLO ---------------------
    def yolo_callback(self, msg: Detection2DArray):
        detections = msg.detections
        obstacle_detected = len(detections) > 0

        out = Bool()
        out.data = obstacle_detected
        self.obstacle_pub.publish(out)

        if obstacle_detected:
            self.get_logger().debug(
                f'[DAY] YOLO detected {len(detections)} objects -> obstacle=True'
            )
        else:
            self.get_logger().debug('[DAY] No YOLO detection -> obstacle=False')

    # -------------------- NIGHT MODE: DEPTH -------------------
    def depth_callback(self, msg: Image):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'Failed to convert depth image: {e}')
            return

        import numpy as np

        depth = depth_image.astype(np.float32)
        h, w = depth.shape[:2]

        # 중앙 ROI (가로/세로 각각 1/3)
        h_start = h // 3
        h_end = 2 * h // 3
        w_start = w // 3
        w_end = 2 * w // 3

        roi = depth[h_start:h_end, w_start:w_end]

        roi_valid = roi[np.isfinite(roi)]
        roi_valid = roi_valid[roi_valid > 0.0]

        if roi_valid.size == 0:
            obstacle_detected = False
            min_dist = math.inf
        else:
            min_dist = float(np.min(roi_valid))
            obstacle_detected = min_dist <= self.max_detection_distance

        out = Bool()
        out.data = obstacle_detected
        self.obstacle_pub.publish(out)

        if obstacle_detected:
            self.get_logger().debug(
                f'[NIGHT] Obstacle detected. min_dist={min_dist:.2f} m '
                f'(threshold={self.max_detection_distance} m)'
            )
        else:
            self.get_logger().debug(
                f'[NIGHT] No obstacle. min_dist={min_dist:.2f} m '
                f'(threshold={self.max_detection_distance} m)'
            )


def main(args=None):
    rclpy.init(args=args)
    node = ObjectPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('ObjectPerceptionNode shutting down...')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

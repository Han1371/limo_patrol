#!/usr/bin/env python3
import time
from collections import deque
from pathlib import Path

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge

import requests
import cv2


class ObjectPerceptionNode(Node):
    """
    - /patrol/cycle_done(True)마다 모드 재결정
    - ✅ 일반 규칙:
        illumination > threshold -> day
        else -> night

    DAY:
      YOLO 감지 -> pause(True) -> one-shot(정지 대기 후 사진+텔레그램) -> pause(False)

    NIGHT:
      depth 근접 감지 -> pause(True) -> 장애물 바라보기 회전(cmd_vel)
      - 거리 증가 추세가 크면 "사람" 가정 -> 텔레그램 메시지 (기본은 pause 유지)
      - 변화폭 작으면 "정지 장애물" -> 짧게 hold 후 pause(False)로 재개
        (옵션 A: Nav2 costmap이 LiDAR /scan으로 회피한다는 전제)
    """

    def __init__(self):
        super().__init__('object_perception_node')

        # ---------- mode params ----------
        self.declare_parameter('illumination', 50.0)
        self.declare_parameter('illumination_threshold', 100.0)

        # ---------- topics ----------
        self.declare_parameter('yolo_topic', '/yolov5/detections')
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')

        self.declare_parameter('pause_topic', '/patrol/pause')
        self.declare_parameter('cycle_done_topic', '/patrol/cycle_done')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        # ---------- day behavior ----------
        self.declare_parameter('day_pause_seconds', 1.0)
        self.declare_parameter('day_cooldown_seconds', 10.0)

        # ---------- night behavior ----------
        self.declare_parameter('max_detection_distance', 2.0)

        self.declare_parameter('roi_y_start_ratio', 0.35)
        self.declare_parameter('roi_y_end_ratio', 0.80)
        self.declare_parameter('roi_x_start_ratio', 0.33)
        self.declare_parameter('roi_x_end_ratio', 0.67)

        self.declare_parameter('distance_percentile', 5.0)
        self.declare_parameter('near_band_m', 0.10)

        self.declare_parameter('turn_kp', 1.8)
        self.declare_parameter('turn_max_w', 0.9)
        self.declare_parameter('center_tolerance_px', 20)

        self.declare_parameter('trend_window_sec', 1.5)
        self.declare_parameter('moving_increase_m', 0.25)
        self.declare_parameter('static_band_m', 0.12)

        self.declare_parameter('night_hold_then_resume_sec', 1.0)
        self.declare_parameter('person_alert_cooldown_sec', 10.0)

        # ---------- Telegram ----------
        self.declare_parameter('telegram_bot_token', '')
        self.declare_parameter('telegram_chat_id', '')

        # ---------- read params ----------
        self.yolo_topic = self.get_parameter('yolo_topic').value
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value

        self.pause_topic = self.get_parameter('pause_topic').value
        self.cycle_done_topic = self.get_parameter('cycle_done_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        self.day_pause_seconds = float(self.get_parameter('day_pause_seconds').value)
        self.day_cooldown = float(self.get_parameter('day_cooldown_seconds').value)

        self.max_detection_distance = float(self.get_parameter('max_detection_distance').value)

        self.roi_y0 = float(self.get_parameter('roi_y_start_ratio').value)
        self.roi_y1 = float(self.get_parameter('roi_y_end_ratio').value)
        self.roi_x0 = float(self.get_parameter('roi_x_start_ratio').value)
        self.roi_x1 = float(self.get_parameter('roi_x_end_ratio').value)

        self.dist_percentile = float(self.get_parameter('distance_percentile').value)
        self.near_band = float(self.get_parameter('near_band_m').value)

        self.turn_kp = float(self.get_parameter('turn_kp').value)
        self.turn_max_w = float(self.get_parameter('turn_max_w').value)
        self.center_tol = int(self.get_parameter('center_tolerance_px').value)

        self.trend_window = float(self.get_parameter('trend_window_sec').value)
        self.moving_increase = float(self.get_parameter('moving_increase_m').value)
        self.static_band = float(self.get_parameter('static_band_m').value)

        self.night_hold_then_resume = float(self.get_parameter('night_hold_then_resume_sec').value)
        self.person_alert_cooldown = float(self.get_parameter('person_alert_cooldown_sec').value)

        self.bot_token = self.get_parameter('telegram_bot_token').value
        self.chat_id = self.get_parameter('telegram_chat_id').value

        # ---------- pubs ----------
        self.mode_pub = self.create_publisher(String, '/patrol/mode', 10)
        self.obstacle_pub = self.create_publisher(Bool, '/patrol/obstacle_detected', 10)
        self.pause_pub = self.create_publisher(Bool, self.pause_topic, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # ---------- subs (dynamic) ----------
        self.yolo_sub = None
        self.rgb_sub = None
        self.depth_sub = None

        self.create_subscription(Bool, self.cycle_done_topic, self.cycle_done_callback, 10)

        self.bridge = CvBridge()
        self.latest_rgb = None

        # DAY state
        self.day_in_action = False
        self.day_last_trigger_time = 0.0
        self.day_one_shot_timer = None

        # NIGHT state
        self.night_paused = False
        self.dist_hist = deque()
        self.night_resume_timer = None
        self.last_person_alert_time = 0.0

        # init mode + subscriptions
        self.mode = self.decide_mode()
        self.apply_mode(self.mode, first=True)
        self.get_logger().info(f'ObjectPerceptionNode started -> mode={self.mode}')

    # ---------------- mode logic ----------------
    def decide_mode(self) -> str:
        illumination = float(self.get_parameter('illumination').value)
        threshold = float(self.get_parameter('illumination_threshold').value)
        # ✅ 일반 규칙
        return 'day' if illumination > threshold else 'night'

    def publish_mode(self):
        self.mode_pub.publish(String(data=self.mode))

    def clear_subs(self):
        if self.yolo_sub is not None:
            self.destroy_subscription(self.yolo_sub)
            self.yolo_sub = None
        if self.rgb_sub is not None:
            self.destroy_subscription(self.rgb_sub)
            self.rgb_sub = None
        if self.depth_sub is not None:
            self.destroy_subscription(self.depth_sub)
            self.depth_sub = None

    def reset_state(self):
        self.publish_pause(False)
        self.stop_cmd()

        self.day_in_action = False
        self.latest_rgb = None
        if self.day_one_shot_timer is not None:
            try:
                self.day_one_shot_timer.cancel()
            except Exception:
                pass
            self.day_one_shot_timer = None

        self.night_paused = False
        self.dist_hist.clear()
        if self.night_resume_timer is not None:
            try:
                self.night_resume_timer.cancel()
            except Exception:
                pass
            self.night_resume_timer = None

    def apply_mode(self, new_mode: str, first: bool = False):
        if (not first) and new_mode == self.mode:
            self.publish_mode()
            return

        self.mode = new_mode
        self.clear_subs()
        self.reset_state()
        self.publish_mode()

        if self.mode == 'day':
            self.yolo_sub = self.create_subscription(Detection2DArray, self.yolo_topic, self.yolo_callback, 10)
            self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 10)
            self.get_logger().warn(f'[MODE] DAY. YOLO={self.yolo_topic}, RGB={self.rgb_topic}')
        else:
            self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
            self.get_logger().warn(f'[MODE] NIGHT. DEPTH={self.depth_topic}')

    def cycle_done_callback(self, msg: Bool):
        if msg.data:
            self.apply_mode(self.decide_mode())

    # ---------------- common helpers ----------------
    def publish_pause(self, pause: bool):
        self.pause_pub.publish(Bool(data=pause))

    def stop_cmd(self):
        t = Twist()
        t.linear.x = 0.0
        t.angular.z = 0.0
        self.cmd_pub.publish(t)

    def send_telegram_text(self, text: str) -> bool:
        if not self.bot_token or not self.chat_id:
            self.get_logger().error('Telegram token/chat_id is empty.')
            return False
        url = f'https://api.telegram.org/bot{self.bot_token}/sendMessage'
        try:
            resp = requests.post(url, data={'chat_id': self.chat_id, 'text': text}, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            self.get_logger().error(f'Telegram text exception: {e}')
            return False

    def send_telegram_photo(self, image_path: str, caption: str = '') -> bool:
        if not self.bot_token or not self.chat_id:
            self.get_logger().error('Telegram token/chat_id is empty.')
            return False
        url = f'https://api.telegram.org/bot{self.bot_token}/sendPhoto'
        try:
            with open(image_path, 'rb') as f:
                files = {'photo': f}
                data = {'chat_id': self.chat_id, 'caption': caption}
                resp = requests.post(url, data=data, files=files, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            self.get_logger().error(f'Telegram photo exception: {e}')
            return False

    # ---------------- DAY ----------------
    def rgb_callback(self, msg: Image):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'RGB convert failed: {e}')

    def yolo_callback(self, msg: Detection2DArray):
        detected = len(msg.detections) > 0
        self.obstacle_pub.publish(Bool(data=detected))
        if not detected:
            return

        now = time.time()
        if self.day_in_action:
            return
        if (now - self.day_last_trigger_time) < self.day_cooldown:
            return

        self.day_in_action = True
        self.day_last_trigger_time = now

        self.get_logger().warn('[DAY] YOLO detected -> pause, capture, telegram, resume')
        self.publish_pause(True)

        # ✅ one-shot: day_pause_seconds 후 딱 1번 실행
        if self.day_one_shot_timer is not None:
            try:
                self.day_one_shot_timer.cancel()
            except Exception:
                pass
        self.day_one_shot_timer = self.create_timer(self.day_pause_seconds, self._day_finish_once)

    def _day_finish_once(self):
        # one-shot cancel
        if self.day_one_shot_timer is not None:
            try:
                self.day_one_shot_timer.cancel()
            except Exception:
                pass
            self.day_one_shot_timer = None

        img_path = self._save_latest_rgb()
        if img_path:
            self.send_telegram_photo(img_path, caption='⚠️ [LIMO][DAY] 사람/장애물 감지 (YOLO)')
        else:
            self.send_telegram_text('⚠️ [LIMO][DAY] 사람/장애물 감지 (YOLO) - 사진 캡처 실패')

        self.publish_pause(False)
        self.day_in_action = False

    def _save_latest_rgb(self) -> str:
        if self.latest_rgb is None:
            return ''
        out_dir = Path('/tmp/limo_patrol_captures')
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        out_path = out_dir / f'day_{ts}.jpg'
        try:
            cv2.imwrite(str(out_path), self.latest_rgb)
            return str(out_path)
        except Exception as e:
            self.get_logger().error(f'Failed to save rgb: {e}')
            return ''

    # ---------------- NIGHT ----------------
    def depth_callback(self, msg: Image):
        import numpy as np

        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'Depth convert failed: {e}')
            return

        depth = depth_img.astype(np.float32)
        h, w = depth.shape[:2]

        # ROI
        y0 = max(0, min(h - 1, int(h * self.roi_y0)))
        y1 = max(1, min(h,     int(h * self.roi_y1)))
        x0 = max(0, min(w - 1, int(w * self.roi_x0)))
        x1 = max(1, min(w,     int(w * self.roi_x1)))

        roi = depth[y0:y1, x0:x1]
        roi_valid = roi[np.isfinite(roi)]
        roi_valid = roi_valid[roi_valid > 0.0]

        if roi_valid.size == 0:
            self.obstacle_pub.publish(Bool(data=False))
            return

        dist = float(np.percentile(roi_valid, self.dist_percentile))
        obstacle_now = dist <= self.max_detection_distance
        self.obstacle_pub.publish(Bool(data=obstacle_now))

        if not obstacle_now:
            return

        # obstacle 발견 -> pause
        if not self.night_paused:
            self.get_logger().warn('[NIGHT] Obstacle detected -> pause and face it')
            self.publish_pause(True)
            self.night_paused = True
            self.dist_hist.clear()

        # pause 중일 때만 cmd_vel로 "바라보기" (Nav2 cmd_vel 충돌 최소화)
        cx = self._estimate_obstacle_cx(roi, x0, x1, dist)
        self._turn_to_center_if_needed(cx, w)

        verdict = self._night_verdict(dist)

        if verdict == 'person':
            now = time.time()
            if (now - self.last_person_alert_time) >= self.person_alert_cooldown:
                self.last_person_alert_time = now
                self.send_telegram_text('🚶 [LIMO][NIGHT] 움직이는 물체 감지 → 사람으로 가정 (depth trend)')
            # 사람일 땐 안전상 정지 유지
            return

        if verdict == 'static_obstacle':
            # 옵션 A: costmap(/scan)이 회피 담당 -> 잠깐 hold 후 resume
            if self.night_resume_timer is None:
                self.get_logger().warn('[NIGHT] Static obstacle -> hold then resume (LiDAR costmap should avoid)')
                self.night_resume_timer = self.create_timer(self.night_hold_then_resume, self._night_resume_once)

    def _night_resume_once(self):
        if self.night_resume_timer is not None:
            try:
                self.night_resume_timer.cancel()
            except Exception:
                pass
            self.night_resume_timer = None

        self.stop_cmd()
        self.publish_pause(False)
        self.night_paused = False
        self.dist_hist.clear()

    def _estimate_obstacle_cx(self, roi, x0: int, x1: int, dist: float) -> float:
        import numpy as np
        band_lo = dist
        band_hi = dist + self.near_band
        mask = (roi >= band_lo) & (roi <= band_hi) & np.isfinite(roi) & (roi > 0.0)

        if np.any(mask):
            ys, xs = np.where(mask)
            return float(np.mean(xs + x0))
        return float((x0 + x1) / 2.0)

    def _turn_to_center_if_needed(self, center_x: float, img_w: int):
        target = img_w / 2.0
        err_px = center_x - target

        if abs(err_px) <= self.center_tol:
            self.stop_cmd()
            return

        err = err_px / (img_w / 2.0)
        w_cmd = -self.turn_kp * err
        w_cmd = max(-self.turn_max_w, min(self.turn_max_w, w_cmd))

        t = Twist()
        t.linear.x = 0.0
        t.angular.z = float(w_cmd)
        self.cmd_pub.publish(t)

    def _night_verdict(self, current_dist: float) -> str:
        now = time.time()
        self.dist_hist.append((now, current_dist))
        while self.dist_hist and (now - self.dist_hist[0][0] > self.trend_window):
            self.dist_hist.popleft()

        if len(self.dist_hist) < 5:
            return 'unknown'

        d0 = self.dist_hist[0][1]
        d1 = self.dist_hist[-1][1]
        increase = d1 - d0

        dmin = min(d for _, d in self.dist_hist)
        dmax = max(d for _, d in self.dist_hist)
        band = dmax - dmin

        if increase >= self.moving_increase:
            return 'person'
        if band <= self.static_band:
            return 'static_obstacle'
        return 'unknown'


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

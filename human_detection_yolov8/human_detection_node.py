#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
from ultralytics import YOLO

class HumanDetectionNode(Node):
    def __init__(self):
        super().__init__('human_detection_node')
        self.get_logger().info("Human Detection Node (ROS2) 起動")

        # カラー画像と深度画像のサブスクリプション
        self.subscription_color = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.color_callback,
            10)
        self.subscription_depth = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10)

        self.bridge = CvBridge()
        # YOLOv8 のモデル（例: yolov8n.pt）を読み込み
        self.yolo_model = YOLO('yolov8n.pt')

        # 最新のメッセージを保持するための変数
        self.latest_color_msg = None
        self.latest_depth_msg = None

        # タイマーにより定期的にメッセージが両方揃っているか確認し処理する
        self.timer = self.create_timer(0.1, self.timer_callback)

    def color_callback(self, msg):
        self.latest_color_msg = msg

    def depth_callback(self, msg):
        self.latest_depth_msg = msg

    def timer_callback(self):
        # 両方のメッセージが取得できていなければ処理しない
        if self.latest_color_msg is None or self.latest_depth_msg is None:
            return

        # ※ ここでは簡易的に最新の2つのメッセージを処理していますが、
        #   タイムスタンプを比較して同期を取る実装に変更することも検討してください。

        try:
            # カラー画像を OpenCV の BGR 画像に変換
            color_image = self.bridge.imgmsg_to_cv2(self.latest_color_msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"カラー画像変換失敗: {e}")
            return

        try:
            # 深度画像を OpenCV の画像に変換 (32FC1: 単位はメートル)
            depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth_msg, '32FC1')
        except Exception as e:
            self.get_logger().error(f"深度画像変換失敗: {e}")
            return

        # YOLOv8 による物体検出（カラー画像を入力）
        results = self.yolo_model(color_image)
        for result in results:
            # 検出されたバウンディングボックス情報を取得
            # 各ボックスは [x1, y1, x2, y2, confidence, class_id] の順になっています
            boxes = result.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                # COCO の "person" はクラスID 0
                if int(cls) == 0:
                    # バウンディングボックスの中心座標
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # 深度画像上の中心座標の深度値を取得
                    if 0 <= cx < depth_image.shape[1] and 0 <= cy < depth_image.shape[0]:
                        distance = depth_image[cy, cx]
                        # 2m以内であれば描画
                        if 0.0 < distance < 2.0:
                            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(color_image, "人検知", (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            self.get_logger().info(f"人検知: 距離 = {distance:.2f} m")

        # 検出結果をウィンドウに表示
        cv2.imshow("YOLOv8 Detection", color_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = HumanDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt により終了")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

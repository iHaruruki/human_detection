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

        # サブスクライバの作成（カラー画像と深度画像）
        self.sub_color = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.color_callback,
            10
        )
        self.sub_depth = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        self.bridge = CvBridge()
        # YOLOv8 モデルのロード（例: yolov8n.pt を使用）
        self.yolo_model = YOLO('yolov8n.pt')

        # 最新の画像メッセージを保持する変数
        self.latest_color_msg = None
        self.latest_depth_msg = None

        # タイマーで定期的に画像処理を実行（約10Hz）
        self.timer = self.create_timer(0.1, self.timer_callback)

    def color_callback(self, msg):
        self.latest_color_msg = msg

    def depth_callback(self, msg):
        self.latest_depth_msg = msg

    def timer_callback(self):
        # 両方の画像がまだ取得できていなければ処理しない
        if self.latest_color_msg is None or self.latest_depth_msg is None:
            return

        try:
            # ROS のカラー画像メッセージを OpenCV の BGR 画像に変換
            color_image = self.bridge.imgmsg_to_cv2(self.latest_color_msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"カラー画像変換エラー: {e}")
            return

        try:
            # ROS の深度画像メッセージを OpenCV の画像に変換 (エンコーディング "32FC1": 単位はメートル)
            depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth_msg, '32FC1')
        except Exception as e:
            self.get_logger().error(f"深度画像変換エラー: {e}")
            return

        # YOLOv8 による物体検出（カラー画像を入力）
        results = self.yolo_model(color_image)
        for result in results:
            # result.boxes.data は [x1, y1, x2, y2, confidence, class] の各情報を保持
            boxes = result.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                # COCO データセットにおける "person" はクラスID 0
                if int(cls) == 0:
                    # バウンディングボックスの中心座標を算出
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # 深度画像のサイズ内であることを確認して、中心の深度値を取得
                    if 0 <= cx < depth_image.shape[1] and 0 <= cy < depth_image.shape[0]:
                        distance = depth_image[cy, cx]
                        # 取得した距離を画像上に表示するテキストに整形（小数点2桁）
                        text = f"人検知: {distance:.2f}mm"
                        # バウンディングボックスの描画
                        cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        # 画像上に距離を表示
                        cv2.putText(color_image, text, (int(x1), int(y1)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        self.get_logger().info(text)

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

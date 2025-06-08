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

        # 表示ウィンドウの設定（ウィンドウサイズを大きくする）
        cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLOv8 Detection", 1280, 720)

        # タイマーで定期的に画像処理を実行（約10Hz）
        self.timer = self.create_timer(0.1, self.timer_callback)

    def color_callback(self, msg):
        self.latest_color_msg = msg

    def depth_callback(self, msg):
        self.latest_depth_msg = msg

    def timer_callback(self):
        # 両方の画像が取得できていなければ処理しない
        if self.latest_color_msg is None or self.latest_depth_msg is None:
            return

        try:
            # カラー画像を OpenCV の BGR 画像に変換
            color_image = self.bridge.imgmsg_to_cv2(self.latest_color_msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"カラー画像変換エラー: {e}")
            return

        try:
            # 深度画像を OpenCV の画像に変換（ここでは "16UC1" で取得）
            depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth_msg, '16UC1')
        except Exception as e:
            self.get_logger().error(f"深度画像変換エラー: {e}")
            return

        # YOLOv8 による物体検出（カラー画像を入力）
        results = self.yolo_model(color_image)
        for result in results:
            boxes = result.boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                if int(cls) == 0:  # クラスID 0 -> person
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    if 0 <= cx < depth_image.shape[1] and 0 <= cy < depth_image.shape[0]:
                        # 深度画像の値はミリメートル単位なので、メートルに変換
                        distance_raw = depth_image[cy, cx]
                        distance = distance_raw / 1000.0  # 例: 600 -> 0.6m

                        text = f"person: {distance:.2f}m"
                        cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(color_image, text, (int(x1), int(y1)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        self.get_logger().info(text)

        # 画像自体を拡大して表示（ここで画像全体を大きくする）
        scale_factor = 2.0  # 例として2倍に拡大
        resized_image = cv2.resize(color_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("YOLOv8 Detection", resized_image)
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

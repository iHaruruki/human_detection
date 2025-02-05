#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
from ultralytics import YOLO

# ========== YOLOv8 モデルをロード ==========
model = YOLO('yolov8n.pt')  # 小型モデル

# ROSメッセージをOpenCVの画像に変換するためのCvBridge
bridge = CvBridge()

def callback(color_msg, depth_msg):
    """
    カラー画像と深度画像のROSメッセージを受け取り、人検知を行う。
    """
    try:
        # ROSのカラー画像をOpenCVの画像 (BGR) に変換
        color_image = bridge.imgmsg_to_cv2(color_msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error (color): %s", e)
        return

    try:
        # ROSの深度画像をOpenCVの画像に変換 (単位はメートル)
        depth_image = bridge.imgmsg_to_cv2(depth_msg, "32FC1")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error (depth): %s", e)
        return

    # YOLOv8 による物体検出（カラー画像を入力）
    results = model(color_image)

    # 各検出結果を処理
    for result in results:
        # 検出されたボックスの情報を取得 ([x1, y1, x2, y2, confidence, class_id])
        boxes = result.boxes.data.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box

            # クラスID 0 は "person"（COCOデータセット）
            if int(cls) == 0:
                # バウンディングボックスの中心座標を計算
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # 深度値を取得（単位: メートル）
                if 0 <= cx < depth_image.shape[1] and 0 <= cy < depth_image.shape[0]:
                    distance = depth_image[cy, cx]

                    # 2m以内なら「人検知」と表示
                    if 0.0 < distance < 2.0:
                        cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(color_image, "人検知", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        rospy.loginfo("人検知: 距離 = {:.2f} m".format(distance))

    # 検出結果を表示
    cv2.imshow("YOLOv8 Detection", color_image)
    cv2.waitKey(1)

def main():
    """
    ROSノードのメイン関数
    """
    rospy.init_node('yolov8_astra_detector', anonymous=True)

    # 画像トピックのサブスクライバを作成
    color_sub = Subscriber('/camera/color/image_raw', Image)
    depth_sub = Subscriber('/camera/depth/image_raw', Image)

    # 2つの画像ストリームを同期させる
    ats = ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=10, slop=0.1)
    ats.registerCallback(callback)

    rospy.loginfo("YOLOv8 Astra Pro 人検知 ノード開始")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("ノードを終了します")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

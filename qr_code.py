#!/usr/bin/env python3

"""
File: qr_code.py
Author: UnnamedOrange
Brief: Detect a QR code and retrieve its pose in the camera frame.
Version: 0.1
Date: 2024-8-1

Copyright: Copyright (c) Alpheus. Licensed under the MIT Licence.
See the LICENSE file in the repository root for full licence text.
"""

import cv2
import rospy

import numpy as np

from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from pyzbar import pyzbar
from sensor_msgs.msg import CameraInfo, Image
from sensors.msg import QrCode
from tf.transformations import quaternion_from_matrix


class MyApplication:
    def camera_info_callback(self, msg):
        self.D = np.array(msg.D, np.float32)
        self.K = np.array(msg.K, np.float32).reshape((3, 3))

    def image_callback(self, msg):
        if self.D is None or self.K is None:
            rospy.logwarn("Camera info has not been retrieved")
            return

        now = rospy.Time.now()
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        detector = cv2.QRCodeDetector()

        qr_codes = pyzbar.decode(cv_image, [pyzbar.ZBarSymbol.QRCODE])
        if not qr_codes:
            self.pub_marked_image.publish(self.bridge.cv2_to_imgmsg(cv_image))
            return

        # TODO: For now ONLY the first QR code is used.
        code_data, _, _, polygon, code_quality, code_orientation = qr_codes[0]
        if code_orientation != "UP" or code_quality != 1:
            print(qr_codes)
        content = str(code_data, "utf-8")
        # TODO: The order of the points in `polygon` is unknown. Use code_orientation to sort.
        points = np.array(
            [
                [polygon[0].x, polygon[0].y],
                [polygon[1].x, polygon[1].y],
                [polygon[2].x, polygon[2].y],
                [polygon[3].x, polygon[3].y],
            ],
            np.float32,
        ).reshape([1, 4, 2])
        cv2.circle(cv_image, np.array(points[0, 0], np.int32), 9, (255, 0, 0), -1)
        cv2.circle(cv_image, np.array(points[0, 1], np.int32), 9, (0, 255, 0), -1)
        cv2.circle(cv_image, np.array(points[0, 2], np.int32), 9, (0, 0, 255), -1)
        cv2.circle(cv_image, np.array(points[0, 3], np.int32), 9, (255, 255, 0), -1)
        self.pub_marked_image.publish(self.bridge.cv2_to_imgmsg(cv_image))

        # In OpenCV, z-axis is heading out and y-axis is heading up.
        # TODO: Now assume the size is 1x1.
        object_points = np.array(
            [
                [-0.5, 0.5, 0],
                [0.5, 0.5, 0],
                [0.5, -0.5, 0],
                [-0.5, -0.5, 0],
            ],
            np.float32,
        )
        object_points *= 0.179
        result, rvec_cv, tvec_cv = cv2.solvePnP(object_points, points, self.K, self.D)
        if not result:
            rospy.logwarn("Failed to solvePnP")
            return

        # Convert OpenCV convention to ROS convention (x-axis out, z-axis up).
        cv_to_ros = np.array(
            [
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ]
        )
        tvec_ros = cv_to_ros @ tvec_cv
        rotation_matrix_cv, _ = cv2.Rodrigues(rvec_cv)
        rotation_matrix_ros = np.eye(4)
        rotation_matrix_ros[:3, :3] = cv_to_ros @ rotation_matrix_cv @ cv_to_ros.T
        quaternion_ros = quaternion_from_matrix(rotation_matrix_ros)

        # The pose of **camera** in the reference frame.
        pose = Pose()
        pose.position.x = tvec_ros[0]
        pose.position.y = tvec_ros[1]
        pose.position.z = tvec_ros[2]
        pose.orientation.x = quaternion_ros[0]
        pose.orientation.y = quaternion_ros[1]
        pose.orientation.z = quaternion_ros[2]
        pose.orientation.w = quaternion_ros[3]

        qr_code = QrCode()
        qr_code.content = content
        qr_code.pose.header.stamp = now
        qr_code.pose.header.frame_id = self.frame_id
        qr_code.pose.pose = pose

        self.pub_qr_code.publish(qr_code)

    def __init__(self):
        rospy.init_node("qr_code", anonymous=True)

        self.image_topic_name = rospy.get_param("~image_topic", None)
        if not self.image_topic_name:
            rospy.logfatal(
                "Topic name of image (image_topic) for QR code is not provided"
            )
            exit(1)

        self.camera_info_topic_name = rospy.get_param("~camera_info_topic", None)
        if not self.camera_info_topic_name:
            rospy.logfatal(
                "Topic name of camera info (camera_info_topic) for QR code is not provided"
            )
            exit(1)

        self.frame_id = rospy.get_param("~frame_id", None)
        if not self.camera_info_topic_name:
            rospy.logfatal(
                "Topic name of frame id (frame_id) for QR code is not provided"
            )
            exit(1)

        self.bridge = CvBridge()
        self.D = None
        self.K = None

        self.pub_qr_code = rospy.Publisher("~qr_code", QrCode, queue_size=128)
        self.pub_marked_image = rospy.Publisher("~marked_image", Image, queue_size=1)

        self.sub_camera_info = rospy.Subscriber(
            self.camera_info_topic_name, CameraInfo, self.camera_info_callback
        )
        self.sub_image = rospy.Subscriber(
            self.image_topic_name, Image, self.image_callback
        )

        rospy.spin()


if __name__ == "__main__":
    MyApplication()

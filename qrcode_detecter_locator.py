import cv2
from pyzbar.pyzbar import decode
import numpy as np
"""
File: qrcode_detecter_locator.py
Author:CloudFan
Brief: Detect a QR code and retrieve its pose in the camera frame.
Version: 0.1
Date: 2024-8-18
"""

class QRCodeDetectorAndLocator:
    def __init__(self, camera_matrix, dist_coeffs):
        self.qr_codes = []
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def detect_qr_codes(self, frame):
        self.qr_codes = decode(frame)
        for obj in self.qr_codes:
            print("Detected QR Code with data:", obj.data.decode("utf-8"))
        
        # 可以在这里绘制边框和显示信息等
        for obj in self.qr_codes:
            points = obj.polygon
            if len(points) > 4:
                hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                points = hull
            n = len(points)
            for j in range(0, n):
                cv2.line(frame, tuple(points[j]), tuple(points[(j + 1) % n]), (255,0,0), 3)
        return frame

    def locate_camera(self):
        if not self.qr_codes:
            print("No QR codes detected.")
            return None

        # 使用第一个二维码
        qr_code = self.qr_codes[0]
        data = qr_code.data.decode("utf-8").split(',')
        world_coords = np.array([float(x) for x in data], dtype=np.float32)

        size = 0.179  # 二维码实际尺寸，单位米
        # 二维码初始化的四个顶点，大小为1x1，初始设置中心点为原点，在XY平面上。后面会根据二维码实际位姿调整
        object_points = np.array([
                [-0.5, 0.5, 0],
                [0.5, 0.5, 0],
                [0.5, -0.5, 0],
                [-0.5, -0.5, 0],
            ],
            np.float32,
        )

        image_points = np.array([point for point in qr_code.polygon], dtype=np.float32)

        # 求解相机位姿
        retval, rvec, tvec = cv2.solvePnP(object_points, image_points, self.camera_matrix, self.dist_coeffs)
        if retval:
            print("Camera located successfully:")
            print("Rotation Vector:", rvec)
            print("Translation Vector:", tvec)
            return rvec, tvec
        else:
            print("Failed to locate camera.")
            return None
    def create_transform_matrix(world_coords, size):
        x, y, z, yaw = world_coords  # 解析world_coords，包含x, y, z位置和偏航角yaw
        yaw = np.deg2rad(yaw)  # 将偏航角从度转换为弧度

        # 创建旋转矩阵，绕Z轴旋转
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # 创建变换矩阵
        T = np.eye(4)
        T[0:3, 0:3] = Rz
        T[0:3, 3] = [x, y, z]

        # 计算世界坐标系中二维码顶点的位置
        # 二维码顶点在本地坐标系中的位置
        local_points = np.array([
            [-0.5 * size, 0.5 * size, 0],
            [0.5 * size, 0.5 * size, 0],
            [0.5 * size, -0.5 * size, 0],
            [-0.5 * size, -0.5 * size, 0]
        ])

        # 将本地坐标变换到世界坐标
        world_points = np.dot(Rz, local_points.T).T + np.array([x, y, z])
        return world_points


# 示例使用
# camera_matrix 和 dist_coeffs 需要从相机标定获取
camera_matrix = np.array(...) 
dist_coeffs = np.array(...)   
detector = QRCodeDetectorAndLocator(camera_matrix, dist_coeffs)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = detector.detect_qr_codes(frame)
    detector.locate_camera()  # 每帧调用
    
    cv2.imshow('QR Code Detection and Localization', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

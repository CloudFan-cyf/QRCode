import cv2
from pyzbar.pyzbar import decode
import numpy as np


def decode_qr_codes(frame):
    decoded_objects = decode(frame)
    for obj in decoded_objects:
        # 打印二维码中的数据
        print("Data:", obj.data.decode("utf-8"))
        # 绘制二维码的边界框
        points = obj.polygon
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            points = hull
        n = len(points)
        for j in range(0, n):
            cv2.line(frame, tuple(points[j]), tuple(points[(j + 1) % n]), (255,0,0), 3)
    return frame

def calculate_camera_pose(frame):
    decoded_objects = decode(frame)
    for obj in decoded_objects:
        # In OpenCV, z-axis is heading out and y-axis is heading up.
        # TODO: Now assume the size is 1x1.
        object_points = np.array([
                [-0.5, 0.5, 0],
                [0.5, 0.5, 0],
                [0.5, -0.5, 0],
                [-0.5, -0.5, 0],
            ],
            np.float32,)
        # 1x1的二维码的实际尺寸大小为0.179米
        object_points *= 0.179
        image_points = np.array([point for point in obj.polygon], dtype=np.float32)
        
        # 使用solvePnP计算相机位姿
        result, rvec, tvec = cv2.solvePnP(object_points, image_points, None, None)
        print("Rotation Vector:\n", rvec)
        print("Translation Vector:\n", tvec)

# 加载本地图片文件
image_path = 'QRCodeDetectTest.jpg'  # 修改为你的图片文件路径
frame = cv2.imread(image_path)

if frame is not None:
    frame = decode_qr_codes(frame)  # 调用之前定义的decode_qr_codes函数
    #calculate_camera_pose(frame)
    cv2.imshow('QR Code Detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error loading image")

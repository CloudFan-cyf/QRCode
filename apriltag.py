import cv2
import os
import numpy as np
import typing
from pupil_apriltags import Detector

def read_image(image_path):
    """读取图像"""
    return cv2.imread(image_path)

def detect_tags(image):
    """检测图像中的apriltag"""
    at_detector = Detector(families='tag25h9',
                           nthreads=4,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=True,
                           decode_sharpening=0.25,
                           debug=False)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return at_detector.detect(gray_image,
                              estimate_tag_pose=True,
                              camera_params=[1355.8792771309486, 1362.066060010168, 951.9092414769614, 526.9399779936876],
                              tag_size=0.4,
                              )

def draw_detections(image, tags):
    """在图像上绘制检测到的标签框架"""
    for tag in tags:
        for idx in range(len(tag.corners)):
            cv2.line(image, tuple(tag.corners[idx - 1].astype(int)), tuple(tag.corners[idx].astype(int)), (0, 255, 0), 2)
        cv2.putText(image, str(tag.tag_id), tuple(tag.corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image

def calculate_camera_world_coordinates(tag_center_world, R, t):
    """计算相机的世界坐标"""
    # 逆旋转和平移来计算相机的世界坐标
    R_inv = np.linalg.inv(R)
    camera_world_coords = tag_center_world - np.dot(R_inv, t)
    return camera_world_coords

def main(image_path, tag_center_world):

    image = read_image(image_path)
    tags = detect_tags(image)

    if tags:
        for tag in tags:
            print(f"Tag ID: {tag.tag_id}")
            print("Rotation Matrix (R):")
            print(tag.pose_R)
            print("Translation Vector (t):")
            print(tag.pose_t)

            # 计算相机的世界坐标
            camera_coords = calculate_camera_world_coordinates(tag_center_world, tag.pose_R, tag.pose_t)
            print("Camera World Coordinates:")
            print(camera_coords)

    # 绘制检测到的标签并展示图像
    image_with_detections = draw_detections(image, tags)
    cv2.imshow('Detected Apriltags', image_with_detections)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 假设标签中心的世界坐标是已知的，例如 [100, 200, 300]
    tag_center_world_coords = np.array([100, 200, 300])
    image_path = 'apriltag_test_2.jpg'  # 修改为你的图像路径
    os.add_dll_directory(r'D:\CYF\anaconda\envs\pyzbar_test\Lib\site-packages\pupil_apriltags.libs')
    main(image_path, tag_center_world_coords)

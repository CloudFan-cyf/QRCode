import cv2
import numpy as np
from pupil_apriltags import Detector
import os

def read_video(video_path):
    return cv2.VideoCapture(video_path)

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def detect_tags(image, at_detector):
    """检测图像中的apriltag"""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return at_detector.detect(gray_image,
                              estimate_tag_pose=True,
                              camera_params=[1355.8792771309486, 1362.066060010168, 951.9092414769614, 526.9399779936876],
                              tag_size=0.4)

def draw_detections(image, tags):
    """在图像上绘制检测到的标签框架"""
    for tag in tags:
        for idx in range(len(tag.corners)):
            cv2.line(image, tuple(tag.corners[idx - 1].astype(int)), tuple(tag.corners[idx].astype(int)), (0, 255, 0), 2)
        cv2.putText(image, str(tag.tag_id), tuple(tag.corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image

def calculate_camera_world_coordinates(tag_center_world, R, t):
    """计算相机的世界坐标"""
    R_inv = np.linalg.inv(R)
    camera_world_coords = tag_center_world - np.dot(R_inv, t)
    return camera_world_coords

def main(video_path, tag_centers_world):
    at_detector = Detector(families='tag25h9',
                           nthreads=4,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=True,
                           decode_sharpening=0.25,
                           debug=False)

    cap = read_video(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
         # 缩放图像，这里设定为原来的50%
        frame = resize_image(frame, 50)

        tags = detect_tags(frame, at_detector)
        for tag in tags:
            if tag.tag_id in tag_centers_world:
                print(f"Tag ID: {tag.tag_id}")
                print("Rotation Matrix (R):")
                print(tag.pose_R)
                print("Translation Vector (t):")
                print(tag.pose_t)

                # 计算相机的世界坐标
                camera_coords = calculate_camera_world_coordinates(tag_centers_world[tag.tag_id], tag.pose_R, tag.pose_t)
                print("Camera World Coordinates:")
                print(camera_coords)

        # 绘制检测到的标签并展示图像
        image_with_detections = draw_detections(frame, tags)
        cv2.imshow('Detected Apriltags', image_with_detections)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    os.add_dll_directory(r'D:\CYF\anaconda\envs\pyzbar_test\Lib\site-packages\pupil_apriltags.libs')
    tag_centers_world = {
        0: np.array([50, 100, 200]),
        2: np.array([100, 200, 300]),
        3: np.array([150, 250, 350])
    }
    video_path = './test_video/movie.mp4'
    main(video_path, tag_centers_world)

   


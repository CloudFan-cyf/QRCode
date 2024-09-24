import cv2
import numpy as np
from pupil_apriltags import Detector
import os
import csv
from scipy.spatial.transform import Rotation as R

camera_matrix = np.array([
    [889.8569361900343, 0,517.5806962153418 ],
    [0, 889.3729751599478, 936.4363669536748],
    [0, 0, 1]
])
dist_coeffs = np.array([-0.008400259546985044, -0.010866888062224318, -0.0022880615900283847, -0.0008785960724191664])

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def undistort_image(image, camera_matrix, dist_coeffs):
    # 校正图像
    undistorted_image = cv2.undistort(
        src=image,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
    )
    return undistorted_image

def read_video(video_path):
    return cv2.VideoCapture(video_path)

def detect_tags(image, at_detector):
    """检测图像中的apriltag"""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return at_detector.detect(gray_image,
                              estimate_tag_pose=True,
                              camera_params=[889.8569361900343, 889.3729751599478, 517.5806962153418, 936.4363669536748],
                              tag_size=0.4)

def draw_detections(image, tags):
    """在图像上绘制检测到的标签框架"""
    for tag in tags:
        for idx in range(len(tag.corners)):
            cv2.line(image, tuple(tag.corners[idx - 1].astype(int)), tuple(tag.corners[idx].astype(int)), (0, 255, 0), 2)
        cv2.putText(image, str(tag.tag_id), tuple(tag.corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image

def calculate_camera_world_pose(tag_center_world, R_world, R, t):
    """计算相机的世界坐标和姿态"""
    R_inv = np.linalg.inv(R)
    t_squeeze = np.squeeze(t)  # 确保t是一个一维数组
    t_inv = -np.dot(R_inv, t_squeeze)
    camera_world_coords = tag_center_world + t_inv
    camera_world_rotation = np.dot(R, R_world.T)
    return camera_world_coords, camera_world_rotation

def average_rotations(rotations):
    """计算一组旋转矩阵的平均旋转"""
    quaternions = [R.from_matrix(rot).as_quat() for rot in rotations]  # 将每个旋转矩阵转换为四元数
    mean_quaternion = np.mean(quaternions, axis=0)  # 计算四元数的平均值
    mean_quaternion /= np.linalg.norm(mean_quaternion)  # 归一化四元数
    return R.from_quat(mean_quaternion).as_matrix()  # 将平均四元数转换回旋转矩阵

def main(video_path, tag_centers_world, tag_rotation_world, output_csv):
    at_detector = Detector(families='tag25h9',
                           nthreads=4,
                           quad_decimate=1.0,
                           quad_sigma=0.1,
                           refine_edges=True,
                           decode_sharpening=0.5,
                           debug=False)

    cap = read_video(video_path)
    frame_counter = 0
    results = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = undistort_image(frame,camera_matrix=camera_matrix,dist_coeffs=dist_coeffs)

        tags = detect_tags(frame, at_detector)
        camera_coordinates = []
        camera_rotations = []

        for tag in tags:
            if tag.tag_id==3:
                camera_pose = calculate_camera_world_pose(tag_centers_world[tag.tag_id], tag_rotation_world[tag.tag_id], tag.pose_R, tag.pose_t)
                camera_coordinates.append(camera_pose[0])
                camera_rotations.append(camera_pose[1])

        if camera_coordinates:
            averaged_position = np.mean(camera_coordinates, axis=0)
            print("Averaged Camera World Coordinates:")
            print(averaged_position)
            averaged_rotation = average_rotations(camera_rotations)
            print("Averaged Camera World Rotation:")
            print(averaged_rotation)
            results.append((frame_counter, *averaged_position, *averaged_rotation))
            #estimated_position = update_kalman_filter(kf, averaged_position)
            #print("Estimated Camera World Coordinates with Kalman Filter:")
            #print(estimated_position)
        frame_counter += 1
        

        # 绘制检测到的标签并展示图像
        image_with_detections = draw_detections(frame, tags)
        image_with_detections = resize_image(image_with_detections, 50)
        image_with_detections = cv2.rotate(image_with_detections, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow('Detected Apriltags', image_with_detections)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

     # 保存结果到CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'X', 'Y', 'Z','Rotation_1', 'Rotation_2', 'Rotation_3'])
        writer.writerows(results)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    os.add_dll_directory(r'D:\CYF\anaconda\envs\pyzbar_test\Lib\site-packages\pupil_apriltags.libs')
    tag_centers_world = {
        2: np.array([3.25,0,-3.39]),
        3: np.array([9.3,1.15,-3.5765])
    }
    tag_rotation_world = {
        2: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # 2号Tag
        3: np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 3号Tag
    }
    video_path = './test_video/movie_moving.mp4'
    output_csv = 'camera_poses_moving.csv'  # 输出CSV文件路径
    main(video_path, tag_centers_world, tag_rotation_world, output_csv)

   


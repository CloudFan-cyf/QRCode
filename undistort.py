import cv2
import numpy as np

# 相机内参矩阵和畸变系数

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

def undistort_omnidirectional(image, camera_matrix, dist_coeffs):
    """ 使用cv2.omnidir进行全向相机畸变校正 """
    # 校正图像
    undistorted_image = cv2.undistort(
        src=image,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
    )
    return undistorted_image

# 使用这个函数处理图像
cap = cv2.VideoCapture('./test_video/movie_static.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 校正图像
    frame = undistort_omnidirectional(frame, camera_matrix, dist_coeffs)
    frame = resize_image(frame, 50)
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imshow('Undistorted Image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

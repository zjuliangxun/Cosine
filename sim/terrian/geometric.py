import cv2
import numpy as np


def draw_rectangle_(src, tar, angle: float = 0, translation=np.array([0, 0])):
    # src = np.random.randint(0, 256, (50, 100), dtype=np.uint8)
    # tar = np.zeros((tar_height, tar_width), dtype=np.uint8)

    tar_height, tar_width = tar.shape

    translation = -np.array([0, tar_height]) + (translation + np.array([0, src.shape[0]]))
    tx, ty = translation[0].item(), -translation[1].item()

    # Get the center of rotation, usually the center of src
    center = (0, src.shape[0])
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1)

    rotation_matrix[0, 2] += tx
    rotation_matrix[1, 2] += ty

    # Apply the transformation and place the result into tar
    # The output size is set to tar's dimensions to automatically handle clipping
    borderValue = 1000
    result = cv2.warpAffine(src, rotation_matrix, (tar_width, tar_height), borderValue=borderValue)
    # tar = np.maximum(tar, result)
    tar[result != borderValue] = result[result != borderValue]


# def plane_normal_and_point(points):
#     """计算通过三点的平面的法向量和一点"""
#     p1, p2, p3 = points
#     # 计算向量
#     v1 = np.array(p2) - np.array(p1)
#     v2 = np.array(p3) - np.array(p1)
#     # 法向量为这两个向量的叉积
#     normal = np.cross(v1, v2)
#     return normal, p1


# def find_line_of_intersection(p1, p2, n1, n2):
#     """通过两个点和两个法向量找到交线的一个点和方向"""
#     # 方向向量为法向量的叉积
#     direction = np.cross(n1, n2)
#     # 解决方程找一个点
#     # 建立方程系 A * [x, y, z] = b
#     A = np.array([n1, n2, direction])
#     b = np.dot(A, np.array([p1, p2, np.zeros(3)]).T)
#     point_on_line = np.linalg.solve(A.T, b)
#     return point_on_line, direction


# # 例子：平面1经过点P1(1,0,0), P2(0,1,0), P3(0,0,1)
# # 平面2经过点Q1(1,1,0), Q2(1,0,1), Q3(0,1,1)
# n1, p1 = plane_normal_and_point([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
# n2, p2 = plane_normal_and_point([(1, 1, 0), (1, 0, 1), (0, 1, 1)])

# point_on_line, direction = find_line_of_intersection(p1, p2, n1, n2)
# print("交线的方向向量:", direction)
# print("交线上的一个点:", point_on_line)

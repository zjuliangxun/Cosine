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

from isaacgym.torch_utils import *
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt


class FrameMap:
    def __init__(self, width, length, offset, vertical_scale, horizontal_scale):

        self.map_scale = np.array([horizontal_scale, horizontal_scale, vertical_scale])

        self.length, self.width = length, width  # y,x
        # np.ceil((self.upper_c - self.lower_c) / self.map_xy_scale).astype(np.int)
        self.map = np.zeros((self.width, self.length), dtype=np.int16)  # NOTE 按照terrain_utils.Terrainl反 一下！

        self.cart2map_offset_c = offset  # 初始时i左下角对齐x，这表示cart点加此平移到map上

        # 父亲坐标系施加后得到本系的，例如把父亲坐标系旋转parent_rot得到本系
        self.parent_trans = np.zeros(3)
        self.parent_rot = 0  # yaw

    def set_coord(self, trans, yaw):
        if isinstance(trans, torch.Tensor):
            trans = trans.cpu().numpy()
        if isinstance(yaw, torch.Tensor):
            yaw = yaw.cpu().numpy()
        self.parent_rot = yaw
        self.parent_trans = trans

    def to_uv(self, coord):
        if isinstance(coord, torch.Tensor):
            coord = coord.cpu().numpy()
        if coord.shape[-1] > 2:
            coord = coord[..., :2]
        uv = np.floor((coord + self.cart2map_offset_c) / self.map_xy_scale).astype(np.int32)
        return self.width - uv[1], uv[0]

    def compose(self, src):  # NOTE 只考虑二维的变换，scale假设都是一样的！
        a = (
            (self.scale_mat_inv @ self.cart2map_mat)
            @ (src.parent_trans_mat @ src.parent_rot_mat_inv)
            @ (src.map2cart_mat @ src.scale_mat)
        )
        return src.parent_rot, a[..., 2].reshape(-1)

    def draw_on_self(self, src_frame, q=None, t=None):
        tar = self.map
        src = src_frame.map
        if q is None or t is None:
            q, t = self.compose(src_frame)
        angle = q  # self._q2angle(q) NOTE
        translation = self._t2trans(t)

        tar_height, tar_width = tar.shape

        translation = -np.array([0, tar_height]) + (translation + np.array([0, src.shape[0]]))
        tx, ty = translation[0].item(), -translation[1].item()
        # Get the center of rotation, usually the center of src
        rotation_matrix = cv2.getRotationMatrix2D((0, src.shape[0]), angle, scale=1)

        rotation_matrix[0, 2] += tx
        rotation_matrix[1, 2] += ty

        # Apply the transformation and place the result into tar
        # The output size is set to tar's dimensions to automatically handle clipping
        borderValue = 1000
        result = cv2.warpAffine(src, rotation_matrix, (tar_width, tar_height), borderValue=borderValue)
        # tar = np.maximum(tar, result)
        tar[result != borderValue] = result[result != borderValue]

    @property
    def height_field_raw(self):
        return self.map

    @property
    def map_xy_scale(self):
        return self.map_scale[:2]

    @property
    def scale_mat(self):
        return np.diag(np.array([self.map_xy_scale[0], self.map_xy_scale[1], 1]))

    @property
    def scale_mat_inv(self):
        return np.diag(1 / np.array([self.map_xy_scale[0], self.map_xy_scale[1], 1]))

    @property
    def map2cart_mat(self):
        a = np.eye(3)
        a[0:2, 2] = -self.cart2map_offset_c
        return a

    @property
    def cart2map_mat(self):
        a = np.eye(3)
        a[0:2, 2] = self.cart2map_offset_c
        return a

    @property
    def parent_trans_mat(self):
        a = np.eye(3)
        a[0:2, 2] = self.parent_trans[0:2]
        return a

    @property
    def parent_rot_mat(self):
        a = cv2.getRotationMatrix2D((0, 0), self.parent_rot, scale=1)  # NOTE 这个是顺时针的
        return np.concatenate([a, np.array([[0, 0, 1]])], axis=0)

    @property
    def parent_rot_mat_inv(self):
        a = cv2.getRotationMatrix2D((0, 0), -self.parent_rot, scale=1)  # NOTE 这个是顺时针的
        return np.concatenate([a, np.array([[0, 0, 1]])], axis=0)

    def _q2angle(self, q):
        raise NotImplementedError

    def _t2trans(self, t):
        return t[:2]

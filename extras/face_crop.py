import cv2
import numpy as np
import os
import torch
import modules.config
from modules.deps_models_download import ensure_facexlib_models

faceRestoreHelper = None


def align_warp_face(self, landmark, border_mode='constant'):
    affine_matrix = cv2.estimateAffinePartial2D(landmark, self.face_template, method=cv2.LMEDS)[0]
    self.affine_matrices.append(affine_matrix)
    if border_mode == 'constant':
        border_mode = cv2.BORDER_CONSTANT
    elif border_mode == 'reflect101':
        border_mode = cv2.BORDER_REFLECT101
    elif border_mode == 'reflect':
        border_mode = cv2.BORDER_REFLECT
    input_img = self.input_img
    cropped_face = cv2.warpAffine(input_img, affine_matrix, self.face_size,
                                  borderMode=border_mode, borderValue=(135, 133, 132))
    return cropped_face


def detect_landmarks_with_threshold(helper, threshold):
    """按指定置信度阈值抽 5 点 landmark。

    为什么不直接用 helper.get_face_landmarks_5()：facexlib 把那一步的阈值**硬编码成 0.97**
    （`extras/facexlib/utils/face_restoration_helper.py:139`），而它本来是为「人脸修复」设计的，
    那里误检代价高。实测这个阈值几乎永远命中不了：

        | 图                      | 0.97 | 0.8 | 0.5 | 0.3 |
        | 真实照片 photo2          |  0   |  0  |  1  |  1  |
        | 动漫生成图（1024x1536）  |  0   |  0  |  0  |  0  |

    连真实照片都要降到 0.5 才检出。所以这里绕过那个函数、直接调检测器，只替换阈值；
    landmark 的抽取方式与上游保持一致（含 template_3points 分支）。
    """
    with torch.no_grad():
        bboxes = helper.face_det.detect_faces(helper.input_img, threshold)

    landmarks = []
    for bbox in bboxes:
        if helper.template_3points:
            landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 11, 2)])
        else:
            landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)])
        landmarks.append(landmark)
    return landmarks


def crop_image(img_rgb, threshold=None):
    """裁出图中最可信的一张脸，供 FaceSwap 的 IP-Adapter 使用。

    threshold 为 None 时沿用 facexlib 的内置阈值（0.97，实测几乎不命中）；给值时按该阈值检测。
    检测不到时返回原图——这是既有设计，意味着整图（包含头发等非人脸特征）会被送进
    IP-Adapter，这也是 FaceSwap 在动漫图上表现为「只注入了发色」的原因。
    """
    global faceRestoreHelper

    ensure_facexlib_models()

    if faceRestoreHelper is None:
        from extras.facexlib.utils.face_restoration_helper import FaceRestoreHelper
        faceRestoreHelper = FaceRestoreHelper(
            upscale_factor=1,
            model_rootpath=modules.config.path_controlnet,
            device='cpu'  # use cpu is safer since we are out of memory management
        )

    faceRestoreHelper.clean_all()
    faceRestoreHelper.read_image(np.ascontiguousarray(img_rgb[:, :, ::-1].copy()))

    if threshold is None:
        faceRestoreHelper.get_face_landmarks_5()
        landmarks = faceRestoreHelper.all_landmarks_5
        used = 0.97
    else:
        landmarks = detect_landmarks_with_threshold(faceRestoreHelper, threshold)
        used = threshold

    # landmarks are already sorted with confidence.

    if len(landmarks) == 0:
        print(f'No face detected (confidence threshold {used})')
        return img_rgb
    else:
        print(f'Detected {len(landmarks)} faces (confidence threshold {used})')

    result = align_warp_face(faceRestoreHelper, landmarks[0])

    return np.ascontiguousarray(result[:, :, ::-1].copy())

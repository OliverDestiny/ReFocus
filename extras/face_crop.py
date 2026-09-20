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
    （`extras/facexlib/utils/face_restoration_helper.py:139`），那是为人脸修复场景定的值，
    对动漫风格命中率很低。实测（检测到的张数）：

实测（faces 检测到的张数）：
        图                0.97  0.6  0.5  0.3  0.2
        真人-证件照         1     1    1    1    1
        真人-全身           1     1    1    2    2
        动漫-全身 A         0     1    1    1    1
        动漫-全身 B         0     0    0    0    1
        动漫-大头 x3        0     0    0    0    0
    真人照片在默认 0.97 下即可检出；动漫是彩票，且大头照在任何阈值下都检不出。
    crop_image 取置信度最高的一张，所以调低阈值不会改变已能检出图像的裁剪结果。

    所以这里绕过那个函数、直接调检测器，只替换阈值；landmark 的抽取方式与上游保持一致
    （含 template_3points 分支）。
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

    threshold 为 None 时沿用 facexlib 的内置阈值（0.97）；给值时按该阈值检测。
    检测不到时返回原图——这是既有设计，意味着整图（包含头发等非人脸特征）会被送进
    IP-Adapter，这也是 FaceSwap 在动漫图上表现为「只注入了发色」的原因。

    多尺度重试：RetinaFace 的 anchor 尺度是固定的，脸在画面里占比过大时会超出最大 anchor，
    反而检不出。实测动漫「大头照」在原尺寸下任何阈值都失败，缩小后能检出（9019 在 0.5、
    7791 在 0.35）。所以由大到小依次试，第一个有命中的尺度即采用。尺度越小越不可靠，
    因此优先用大尺度；同一尺度内取置信度最高的那张。
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

    bgr = np.ascontiguousarray(img_rgb[:, :, ::-1].copy())
    used = 0.97 if threshold is None else threshold

    landmarks = None
    used_scale = 1.0
    for scale in (1.0, 0.75, 0.5, 0.35):
        if scale == 1.0:
            im = bgr
        else:
            h, w = bgr.shape[:2]
            im = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                            interpolation=cv2.INTER_AREA)

        faceRestoreHelper.clean_all()
        faceRestoreHelper.read_image(im)

        if threshold is None:
            faceRestoreHelper.get_face_landmarks_5()
            found = faceRestoreHelper.all_landmarks_5
        else:
            found = detect_landmarks_with_threshold(faceRestoreHelper, threshold)

        if len(found) > 0:
            # 检测在缩放图上做，换算回原图坐标（对齐裁剪要用原图）
            landmarks = found if scale == 1.0 else [lm / scale for lm in found]
            used_scale = scale
            break

    if landmarks is None:
        print(f'No face detected (confidence threshold {used}, scales tried 1.0/0.75/0.5/0.35)')
        return img_rgb

    print(f'Detected {len(landmarks)} faces (confidence threshold {used}, scale {used_scale})')

    faceRestoreHelper.clean_all()
    faceRestoreHelper.read_image(bgr)
    result = align_warp_face(faceRestoreHelper, landmarks[0])

    return np.ascontiguousarray(result[:, :, ::-1].copy())

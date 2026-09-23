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
    """Extract 5-point landmarks at the given confidence threshold, bypassing the hardcoded 0.97."""
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
    """Crop the most confident face for FaceSwap's IP-Adapter; falls back to the original image.
    Tries 4 scales large to small: RetinaFace's fixed anchors miss faces that fill too much of the frame."""
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
            # Detection runs on the scaled image; landmarks are scaled back for the full-size crop
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

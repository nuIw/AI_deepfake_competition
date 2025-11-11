"""
얼굴 검출 및 크롭 모듈

task.ipynb의 얼굴 검출 로직을 모듈화
"""

import cv2
import numpy as np
from PIL import Image
import dlib


def get_boundingbox(face, width, height, scale_factor=1.3):
    """
    얼굴 영역을 기반으로 정사각형 바운딩 박스를 계산합니다.
    
    Args:
        face: dlib.rectangle 객체
        width: 이미지 너비
        height: 이미지 높이
        scale_factor: 얼굴 영역 확장 비율 (기본값: 1.3)
    
    Returns:
        tuple: (x1, y1, size_bb)
    """
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale_factor)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, size_bb


def detect_and_crop_face(
    image: Image.Image,
    face_detector,
    target_size=(256, 256),
    resize_for_detection=640,
    scale_factor=1.3
):
    """
    dlib 탐지기를 사용하여 얼굴을 검출하고 크롭합니다.
    
    Args:
        image: PIL Image
        face_detector: dlib 얼굴 탐지기
        target_size: 출력 이미지 크기
        resize_for_detection: 탐지 시 리사이즈할 최대 너비
        scale_factor: 얼굴 영역 확장 비율 (기본값: 1.3)
    
    Returns:
        PIL Image or None: 크롭된 얼굴 이미지 (실패 시 None)
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    original_np = np.array(image)
    original_h, original_w, _ = original_np.shape
    
    if original_w == 0 or original_h == 0:
        return None
    
    # 큰 이미지는 리사이즈하여 검출 속도 향상
    if original_w > resize_for_detection:
        scale = resize_for_detection / float(original_w)
        resized_h = int(original_h * scale)
        
        if resized_h <= 0 or resize_for_detection <= 0:
            return None
        
        resized_np = cv2.resize(
            original_np,
            (resize_for_detection, resized_h),
            interpolation=cv2.INTER_AREA
        )
    else:
        scale = 1.0
        resized_np = original_np
    
    # 얼굴 검출
    faces = face_detector(resized_np, 1)
    
    if not faces:
        return None
    
    # 가장 큰 얼굴 선택
    face = max(faces, key=lambda rect: rect.width() * rect.height())
    
    # 원본 좌표로 스케일 복원
    scaled_face_rect = dlib.rectangle(
        left=int(face.left() / scale),
        top=int(face.top() / scale),
        right=int(face.right() / scale),
        bottom=int(face.bottom() / scale)
    )
    
    x, y, size = get_boundingbox(scaled_face_rect, original_w, original_h, scale_factor)
    
    if size == 0:
        return None
    
    # 크롭 및 리사이즈
    cropped_np = original_np[y:y + size, x:x + size]
    face_img = Image.fromarray(cropped_np).resize(target_size, Image.BICUBIC)
    
    return face_img


def detect_and_crop_face_batch(
    images,
    face_detector,
    target_size=(256, 256),
    resize_for_detection=640,
    scale_factor=1.3,
    return_original_on_fail=True
):
    """
    여러 이미지에서 얼굴을 검출하고 크롭합니다.
    
    Args:
        images: PIL Image 리스트
        face_detector: dlib 얼굴 탐지기
        target_size: 출력 이미지 크기
        resize_for_detection: 탐지 시 리사이즈할 최대 너비
        scale_factor: 얼굴 영역 확장 비율 (기본값: 1.3)
        return_original_on_fail: 검출 실패 시 원본 이미지 반환 여부
    
    Returns:
        list: 크롭된 이미지 리스트
    """
    results = []
    
    for image in images:
        face_img = detect_and_crop_face(
            image,
            face_detector,
            target_size,
            resize_for_detection,
            scale_factor
        )
        
        if face_img is None and return_original_on_fail:
            # 검출 실패 시 원본 이미지 리사이즈해서 반환
            face_img = image.resize(target_size, Image.BICUBIC)
        
        if face_img is not None:
            results.append(face_img)
    
    return results


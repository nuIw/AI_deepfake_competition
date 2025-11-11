"""
커스텀 Dataset 클래스들

다양한 전처리 방식을 지원하는 Dataset 클래스 제공
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import dlib
import numpy as np

from .face_detection import detect_and_crop_face


class FaceDetectionDataset(Dataset):
    """
    얼굴 검출 기반 Dataset
    
    dlib를 사용하여 얼굴을 검출하고 크롭한 후 학습에 사용합니다.
    ImageFolder와 동일한 구조를 기대합니다: root/class/images
    """
    
    def __init__(
        self,
        root,
        transform=None,
        target_size=(256, 256),
        resize_for_detection=640,
        scale_factor=1.3,
        return_original_on_fail=True,
        face_detector=None
    ):
        """
        Args:
            root: 데이터셋 루트 디렉토리 (class 폴더들이 있는 위치)
            transform: torchvision transforms
            target_size: 얼굴 크롭 후 리사이즈 크기
            resize_for_detection: 검출 전 리사이즈 크기 (속도 향상)
            scale_factor: 얼굴 영역 확장 비율 (기본값: 1.3)
            return_original_on_fail: 얼굴 검출 실패 시 원본 이미지 사용 여부
            face_detector: 외부에서 전달받은 dlib detector (None이면 자동 생성)
        """
        self.root = Path(root)
        self.transform = transform
        self.target_size = target_size
        self.resize_for_detection = resize_for_detection
        self.scale_factor = scale_factor
        self.return_original_on_fail = return_original_on_fail
        
        # dlib 탐지기 (워커별로 한 번만 초기화됨)
        self._face_detector = face_detector
        
        # 이미지 파일 및 라벨 수집
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        
        # 클래스 폴더 탐색
        class_folders = sorted([d for d in self.root.iterdir() if d.is_dir()])
        
        if not class_folders:
            raise ValueError(f"No class folders found in {root}")
        
        self.classes = [d.name for d in class_folders]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 이미지 파일 수집
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for class_folder in class_folders:
            class_idx = self.class_to_idx[class_folder.name]
            
            for img_file in class_folder.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    self.samples.append((str(img_file), class_idx))
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root}")
        
        print(f"FaceDetectionDataset: Loaded {len(self.samples)} images")
        print(f"  Classes: {self.classes}")
        for cls_name, cls_idx in self.class_to_idx.items():
            count = sum(1 for _, label in self.samples if label == cls_idx)
            print(f"    {cls_name} ({cls_idx}): {count} images")
    
    @property
    def face_detector(self):
        """워커별로 한 번만 dlib 탐지기를 초기화합니다."""
        if self._face_detector is None:
            self._face_detector = dlib.get_frontal_face_detector()
        return self._face_detector
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 이미지 로드
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 오류 시 검은 이미지 반환
            image = Image.new('RGB', self.target_size, color='black')
        
        # 얼굴 검출 및 크롭
        face_img = detect_and_crop_face(
            image,
            self.face_detector,
            self.target_size,
            self.resize_for_detection,
            self.scale_factor
        )
        
        # 검출 실패 시 처리
        if face_img is None:
            if self.return_original_on_fail:
                face_img = image.resize(self.target_size, Image.BICUBIC)
            else:
                # 검은 이미지 반환
                face_img = Image.new('RGB', self.target_size, color='black')
        
        # Transform 적용
        if self.transform is not None:
            face_img = self.transform(face_img)
        
        return face_img, torch.tensor(label, dtype=torch.long)


class StandardDataset(Dataset):
    """
    표준 Dataset (ImageFolder와 동일)
    
    얼굴 검출 없이 일반 이미지 처리
    """
    
    def __init__(self, root, transform=None):
        """
        Args:
            root: 데이터셋 루트 디렉토리
            transform: torchvision transforms
        """
        self.root = Path(root)
        self.transform = transform
        
        # 이미지 파일 및 라벨 수집
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        
        # 클래스 폴더 탐색
        class_folders = sorted([d for d in self.root.iterdir() if d.is_dir()])
        
        if not class_folders:
            raise ValueError(f"No class folders found in {root}")
        
        self.classes = [d.name for d in class_folders]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 이미지 파일 수집
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for class_folder in class_folders:
            class_idx = self.class_to_idx[class_folder.name]
            
            for img_file in class_folder.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    self.samples.append((str(img_file), class_idx))
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root}")
        
        print(f"StandardDataset: Loaded {len(self.samples)} images")
        print(f"  Classes: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 이미지 로드
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (256, 256), color='black')
        
        # Transform 적용
        if self.transform is not None:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


class AlbumentationsDataset(Dataset):
    """
    Albumentations를 사용하는 표준 Dataset
    
    Albumentations는 numpy array를 입력으로 받으므로 PIL Image를 변환해야 합니다.
    """
    
    def __init__(self, root, transform=None):
        """
        Args:
            root: 데이터셋 루트 디렉토리
            transform: albumentations.Compose transform
        """
        self.root = Path(root)
        self.transform = transform
        
        # 이미지 파일 및 라벨 수집
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        
        # 클래스 폴더 탐색
        class_folders = sorted([d for d in self.root.iterdir() if d.is_dir()])
        
        if not class_folders:
            raise ValueError(f"No class folders found in {root}")
        
        self.classes = [d.name for d in class_folders]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 이미지 파일 수집
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for class_folder in class_folders:
            class_idx = self.class_to_idx[class_folder.name]
            
            for img_file in class_folder.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    self.samples.append((str(img_file), class_idx))
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root}")
        
        print(f"AlbumentationsDataset: Loaded {len(self.samples)} images")
        print(f"  Classes: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 이미지 로드 (numpy array로 변환)
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)  # albumentations는 numpy array 사용
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Albumentations transform 적용
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, torch.tensor(label, dtype=torch.long)


class AlbumentationsFaceDetectionDataset(Dataset):
    """
    얼굴 검출 + Albumentations를 사용하는 Dataset
    """
    
    def __init__(
        self,
        root,
        transform=None,
        target_size=(256, 256),
        resize_for_detection=640,
        scale_factor=1.3,
        return_original_on_fail=True,
        face_detector=None
    ):
        """
        Args:
            root: 데이터셋 루트 디렉토리
            transform: albumentations.Compose transform
            target_size: 얼굴 크롭 후 리사이즈 크기
            resize_for_detection: 검출 전 리사이즈 크기 (속도 향상)
            scale_factor: 얼굴 영역 확장 비율 (기본값: 1.3)
            return_original_on_fail: 얼굴 검출 실패 시 원본 이미지 사용 여부
            face_detector: 외부에서 전달받은 dlib detector (None이면 자동 생성)
        """
        self.root = Path(root)
        self.transform = transform
        self.target_size = target_size
        self.resize_for_detection = resize_for_detection
        self.scale_factor = scale_factor
        self.return_original_on_fail = return_original_on_fail
        
        # dlib 탐지기
        self._face_detector = face_detector
        
        # 이미지 파일 및 라벨 수집
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        
        # 클래스 폴더 탐색
        class_folders = sorted([d for d in self.root.iterdir() if d.is_dir()])
        
        if not class_folders:
            raise ValueError(f"No class folders found in {root}")
        
        self.classes = [d.name for d in class_folders]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 이미지 파일 수집
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for class_folder in class_folders:
            class_idx = self.class_to_idx[class_folder.name]
            
            for img_file in class_folder.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    self.samples.append((str(img_file), class_idx))
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root}")
        
        print(f"AlbumentationsFaceDetectionDataset: Loaded {len(self.samples)} images")
        print(f"  Classes: {self.classes}")
        for cls_name, cls_idx in self.class_to_idx.items():
            count = sum(1 for _, label in self.samples if label == cls_idx)
            print(f"    {cls_name} ({cls_idx}): {count} images")
    
    @property
    def face_detector(self):
        """워커별로 한 번만 dlib 탐지기를 초기화합니다."""
        if self._face_detector is None:
            self._face_detector = dlib.get_frontal_face_detector()
        return self._face_detector
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 이미지 로드
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', self.target_size, color='black')
        
        # 얼굴 검출 및 크롭
        face_img = detect_and_crop_face(
            image,
            self.face_detector,
            self.target_size,
            self.resize_for_detection,
            self.scale_factor
        )
        
        # 검출 실패 시 처리
        if face_img is None:
            if self.return_original_on_fail:
                face_img = image.resize(self.target_size, Image.BICUBIC)
            else:
                face_img = Image.new('RGB', self.target_size, color='black')
        
        # PIL Image를 numpy array로 변환 (albumentations용)
        face_img = np.array(face_img)
        
        # Albumentations transform 적용
        if self.transform is not None:
            transformed = self.transform(image=face_img)
            face_img = transformed['image']
        
        return face_img, torch.tensor(label, dtype=torch.long)


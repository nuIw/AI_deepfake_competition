# 전처리 모듈 사용 가이드

## 개요

학습과 추론에 다양한 전처리 기법을 적용할 수 있도록 모듈화했습니다.

## 전처리 방식

### 1. 얼굴 검출 기반 (`data_face_detection`)

**특징:**
- dlib를 사용하여 얼굴 검출
- 검출된 얼굴 영역만 크롭하여 학습
- 1.3배 확대된 정사각형 바운딩 박스 사용
- 검출 실패 시 원본 이미지 사용 (선택 가능)

**장점:**
- 얼굴 영역에 집중하여 학습
- 배경 노이즈 제거
- task.ipynb와 동일한 전처리 적용

**사용법:**
```bash
python src/train.py data=data_face_detection
```

**설정:**
```yaml
# configs/data/data_face_detection.yaml
face_detection:
  target_size: [256, 256]           # 최종 크기
  resize_for_detection: 640         # 검출 속도 향상을 위한 리사이즈
  return_original_on_fail: true     # 검출 실패 시 원본 사용
```

### 2. 표준 방식 (`data_standard`)

**특징:**
- 얼굴 검출 없이 전체 이미지 사용
- ImageFolder와 유사하지만 커스텀 Dataset
- 빠른 학습 속도

**장점:**
- 전처리 오버헤드 최소화
- 안정적인 학습

**사용법:**
```bash
python src/train.py data=data_standard
```

### 3. ImageFolder 방식 (기존, `data`)

**특징:**
- torchvision.datasets.ImageFolder 사용
- 가장 단순한 방식

**사용법:**
```bash
python src/train.py data=data
# 또는 (기본값)
python src/train.py
```

## 모듈 구조

```
src/preprocessing/
├── __init__.py           # 모듈 초기화
├── face_detection.py     # 얼굴 검출 함수들
└── datasets.py           # 커스텀 Dataset 클래스들
```

### face_detection.py

**주요 함수:**
- `get_boundingbox(face, width, height, scale_factor)`: 바운딩 박스 계산
- `detect_and_crop_face(image, face_detector, ...)`: 얼굴 검출 및 크롭
- `detect_and_crop_face_batch(images, ...)`: 배치 처리

**파라미터:**
- `scale_factor`: 얼굴 영역 확장 비율 (기본값: 1.3)
  - `1.0`: 검출된 얼굴 영역 그대로 사용
  - `1.3`: 얼굴 영역을 1.3배로 확장 (여유 공간 포함)
  - `1.5`: 더 넓은 영역 포함 (배경 더 많이 포함)

**예시:**
```python
from preprocessing import detect_and_crop_face
import dlib
from PIL import Image

# 얼굴 검출기 초기화
detector = dlib.get_frontal_face_detector()

# 이미지 로드
image = Image.open("image.jpg")

# 얼굴 검출 및 크롭
face_img = detect_and_crop_face(
    image, 
    detector, 
    target_size=(256, 256),
    resize_for_detection=640,
    scale_factor=1.3  # 얼굴 영역 1.3배 확장
)

if face_img is not None:
    face_img.show()
```

### datasets.py

**클래스:**

1. `FaceDetectionDataset`:
   - 얼굴 검출 기반
   - dlib 탐지기를 워커별로 한 번만 초기화 (효율적)
   - 검출 실패 시 원본 이미지 반환 옵션

2. `StandardDataset`:
   - ImageFolder와 유사
   - 얼굴 검출 없음

**예시:**
```python
from preprocessing.datasets import FaceDetectionDataset
from torchvision import transforms
from torch.utils.data import DataLoader

# Transform 정의
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# 데이터셋 생성
dataset = FaceDetectionDataset(
    root="./data/train",
    transform=transform,
    target_size=(256, 256),
    scale_factor=1.3,  # 얼굴 영역 확장 비율
    return_original_on_fail=True
)

# DataLoader 생성
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

## Config 오버라이드

### 얼굴 검출 파라미터 변경
```bash
# target_size 변경
python src/train.py data=data_face_detection \
    data.face_detection.target_size=[224,224]

# scale_factor 변경 (얼굴 영역 확장 비율)
python src/train.py data=data_face_detection \
    data.face_detection.scale_factor=1.5

# 검출 실패 시 건너뛰기
python src/train.py data=data_face_detection \
    data.face_detection.return_original_on_fail=false

# resize_for_detection 변경 (속도 조정)
python src/train.py data=data_face_detection \
    data.face_detection.resize_for_detection=480
```

### Transform 추가
```bash
# 새로운 augmentation 추가
python src/train.py data=data_face_detection \
    '+data.train_loader.dataset.transform.transforms.2={_target_: torchvision.transforms.RandomRotation, degrees: 10}'
```

## 성능 비교

| 방식 | 전처리 시간 | 학습 속도 | 장점 | 단점 |
|------|-----------|---------|------|------|
| **얼굴 검출** | 느림 | 보통 | 얼굴에 집중, 배경 노이즈 제거 | dlib 의존성, 느린 전처리 |
| **표준** | 빠름 | 빠름 | 빠른 학습, 안정적 | 배경 포함 |
| **ImageFolder** | 빠름 | 빠름 | 가장 단순 | 배경 포함 |

## 추가 전처리 기법 구현 예시

새로운 전처리를 추가하려면:

1. `src/preprocessing/` 에 새 모듈 추가
2. `src/preprocessing/__init__.py` 에 import 추가
3. `configs/data/` 에 새 config 파일 추가

**예: MediaPipe 얼굴 검출**
```python
# src/preprocessing/mediapipe_detection.py
import mediapipe as mp

def detect_face_mediapipe(image):
    mp_face_detection = mp.solutions.face_detection
    # ... 구현
    return cropped_face
```

```yaml
# configs/data/data_mediapipe.yaml
train_loader:
  dataset:
    _target_: preprocessing.datasets.MediaPipeDataset
    # ... 설정
```

## 문제 해결

### dlib 설치 오류
```bash
# Windows
pip install dlib

# Linux
sudo apt-get install cmake
pip install dlib

# conda
conda install -c conda-forge dlib
```

### 메모리 부족
- `num_workers` 감소
- `batch_size` 감소
- `resize_for_detection` 값 감소 (640 → 480)

### 얼굴 검출 실패율이 높음
- `return_original_on_fail=True` 설정
- `resize_for_detection` 값 증가 (더 정확한 검출)
- 다른 검출기 사용 (MediaPipe, MTCNN 등)

## 요약

```bash
# 얼굴 검출 기반 학습 (추천: task.ipynb와 동일)
python src/train.py data=data_face_detection

# 표준 방식 (빠른 실험)
python src/train.py data=data_standard

# 기존 방식 (기본값)
python src/train.py
```


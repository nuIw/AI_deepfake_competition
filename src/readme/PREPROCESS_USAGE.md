# 데이터 전처리 가이드

이 문서는 `preprocess.py`를 사용하여 wandb Artifact로부터 raw 데이터를 가져와 전처리하는 방법을 설명합니다.

## 개요

`preprocess.py`는 다음 작업을 수행합니다:
1. wandb에서 raw dataset artifact를 다운로드
2. **train, val, test 각각을 분리하여** 이미지 전처리 (리사이징 등)
3. 전처리된 데이터를 PIL 이미지로 저장 (ImageFolder 호환)
4. 전처리된 데이터를 새로운 wandb artifact로 업로드

### 데이터 구조

**입력 (Raw Artifact):**
```
raw_data/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       ├── img1.jpg
│       └── img2.jpg
├── val/
│   ├── class1/
│   └── class2/
└── test/
    ├── class1/
    └── class2/
```

**출력 (Processed):**
```
processed_data_output/
├── train/
│   ├── class1/
│   │   ├── img1.jpg  (224x224로 리사이즈됨)
│   │   └── img2.jpg
│   └── class2/
├── val/
│   ├── class1/
│   └── class2/
└── test/
    ├── class1/
    └── class2/
```

## 사용 방법

### 1. 기본 실행

```bash
cd src
python preprocess.py
```

### 2. 설정 오버라이드

Hydra를 사용하므로 명령줄에서 설정을 오버라이드할 수 있습니다:

```bash
# 다른 artifact 사용
python preprocess.py artifact.raw_artifact_name="raw-dataset:v1"

# 이미지 크기 변경 (Resize transform의 size 파라미터 변경)
python preprocess.py preprocess.transform.transforms.0.size=[256,256]

# 출력 디렉토리 변경
python preprocess.py preprocess.output_dir="./my_processed_data"

# 여러 설정 동시 변경
python preprocess.py \
  artifact.raw_artifact_name="raw-dataset:v2" \
  preprocess.transform.transforms.0.size=[512,512] \
  artifact.processed_artifact_name="processed-dataset-512"

# train과 val만 전처리 (test 제외)
python preprocess.py preprocess.splits="[train,val]"
```

### 3. 다른 프로젝트/entity 사용

```bash
python preprocess.py \
  wandb.project_name="my-project" \
  wandb.entity="my-team"
```

## 설정 파일

설정은 `configs/preprocess.yaml`에서 관리됩니다:

```yaml
# wandb 설정
wandb:
  project_name: "MIP-0"
  entity: 'dmachine-kyung-hee-university'

# Artifact 관련 설정
artifact:
  raw_artifact_name: "raw-dataset:latest"
  raw_artifact_type: "raw_data"
  processed_artifact_name: "processed-dataset"
  processed_artifact_type: "dataset"
  description: "Preprocessed dataset with resized images"

# 전처리 설정
preprocess:
  output_dir: "./processed_data_output"
  
  # 전처리할 split 목록 (없는 split은 자동으로 스킵)
  splits: ["train", "val", "test"]
  
  # Hydra instantiate를 사용한 Transform 파이프라인
  transform:
    _target_: torchvision.transforms.Compose
    transforms:
      - _target_: torchvision.transforms.Resize
        size: [224, 224]
      - _target_: torchvision.transforms.ToTensor
      - _target_: torchvision.transforms.ToPILImage
```

## 전처리 파이프라인

**Hydra의 `instantiate`를 사용**하여 설정 파일에서 전처리 파이프라인을 관리합니다.

기본 전처리는 다음 단계를 거칩니다:

1. **Resize**: 모든 이미지를 지정된 크기로 리사이징
2. **ToTensor**: PIL 이미지를 텐서로 변환
3. **ToPILImage**: 다시 PIL 이미지로 변환 (저장을 위해)

전처리된 이미지는 PIL 형식으로 저장되어 나중에 `torchvision.datasets.ImageFolder`로 쉽게 로드할 수 있습니다.

### Transform 커스터마이징

`configs/preprocess.yaml`에서 Transform을 자유롭게 추가/수정할 수 있습니다:

```yaml
preprocess:
  transform:
    _target_: torchvision.transforms.Compose
    transforms:
      - _target_: torchvision.transforms.Resize
        size: [256, 256]
      - _target_: torchvision.transforms.Grayscale  # 그레이스케일 추가
      - _target_: torchvision.transforms.ToTensor
      - _target_: torchvision.transforms.Normalize  # 정규화 추가
        mean: [0.5]
        std: [0.5]
      - _target_: torchvision.transforms.ToPILImage
```

## 출력 구조

전처리된 데이터는 **train/val/test를 분리하여** 다음과 같은 구조로 저장됩니다:

```
processed_data_output/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image1.jpg
│       └── image2.jpg
├── val/
│   ├── class1/
│   └── class2/
└── test/
    ├── class1/
    └── class2/
```

각 split(train/val/test) 디렉토리는 `ImageFolder`가 요구하는 형식과 동일합니다.

### train.py에서 사용하기

전처리된 데이터를 학습에 사용할 때는 data.yaml의 path만 변경하면 됩니다:

```bash
python train.py data.path=./processed_data_output
```

또는 data.yaml을 직접 수정:
```yaml
path: "./processed_data_output"
```

## wandb Artifact 버전 관리

### Latest 버전 사용
```bash
python preprocess.py artifact.raw_artifact_name="raw-dataset:latest"
```

### 특정 버전 사용
```bash
python preprocess.py artifact.raw_artifact_name="raw-dataset:v0"
python preprocess.py artifact.raw_artifact_name="raw-dataset:v1"
```

### Artifact alias 사용
```bash
python preprocess.py artifact.raw_artifact_name="raw-dataset:production"
```

## 전체 워크플로우 예제

```bash
# 1. Raw 데이터를 artifact로 업로드 (이미 완료된 경우 스킵)
python raw.py --project MIP-0 --entity dmachine-kyung-hee-university \
  create --name raw-dataset --type raw_data --dir ./raw_data

# 2. 전처리 실행
cd src
python preprocess.py

# 3. 전처리된 데이터로 학습
python train.py data.path=./processed_data_output
```

## 주의사항

1. **PIL 저장 형식**: 전처리된 이미지는 PIL 형식으로 저장되므로 `ImageFolder`로 로드 가능합니다.
2. **디스크 공간**: 전처리된 데이터는 로컬에 저장되므로 충분한 디스크 공간이 필요합니다.
3. **Artifact 이름**: `processed_artifact_name`이 이미 존재하면 새 버전이 생성됩니다.
4. **wandb 로그인**: 실행 전에 `wandb login`으로 로그인이 필요합니다.

## 문제 해결

### wandb 로그인 오류
```bash
wandb login
```

### CUDA out of memory
전처리는 CPU에서 실행되므로 CUDA 메모리 문제는 발생하지 않습니다.

### 디스크 공간 부족
`output_dir`을 충분한 공간이 있는 디렉토리로 변경하세요:
```bash
python preprocess.py preprocess.output_dir="/path/to/large/disk"
```

## 확장 가능성

### 명령줄에서 Transform 추가

Hydra의 오버라이드 기능을 사용하여 명령줄에서 Transform을 추가할 수 있습니다:

```bash
# 명령줄에서 새로운 transform 추가
python preprocess.py \
  '+preprocess.transform.transforms.1={_target_: torchvision.transforms.RandomHorizontalFlip, p: 0.5}'
```

### 설정 파일로 커스텀 파이프라인 생성

더 복잡한 전처리를 위해서는 `configs/preprocess.yaml`을 직접 수정하세요:

```yaml
preprocess:
  transform:
    _target_: torchvision.transforms.Compose
    transforms:
      - _target_: torchvision.transforms.Resize
        size: [512, 512]
      - _target_: torchvision.transforms.CenterCrop
        size: [448, 448]
      - _target_: torchvision.transforms.ColorJitter
        brightness: 0.2
        contrast: 0.2
      - _target_: torchvision.transforms.ToTensor
      - _target_: torchvision.transforms.ToPILImage
```

### 커스텀 Transform 클래스 사용

직접 작성한 Transform 클래스도 사용 가능합니다:

```yaml
preprocess:
  transform:
    _target_: torchvision.transforms.Compose
    transforms:
      - _target_: my_custom_transforms.CustomTransform
        param1: value1
        param2: value2
      - _target_: torchvision.transforms.ToTensor
      - _target_: torchvision.transforms.ToPILImage
```


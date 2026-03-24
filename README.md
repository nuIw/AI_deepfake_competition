<div align="center">

# 🕵️ Deepfake Detection Challenge

**딥페이크 범죄 대응을 위한 AI 탐지 모델 경진대회**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![WandB](https://img.shields.io/badge/Logging-WandB-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black)](https://wandb.ai/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-FFD21E?style=flat-square)](https://huggingface.co/)

이미지 및 동영상 프레임을 입력으로 받아 **실제(Real) / 가짜(Fake)** 를 구분하는 딥페이크 탐지 모델  
평가 지표: **Macro F1-score** | 최종 성적: **0.8194**

</div>

---

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [데이터셋](#-데이터셋)
- [전처리](#-전처리)
- [모델 아키텍처](#-모델-아키텍처)
- [실험 및 결과](#-실험-및-결과)
- [팀 역할 분담](#-팀-역할-분담)
- [한계점 및 향후 개선 방향](#-한계점-및-향후-개선-방향)
- [결론](#-결론)
- [참고문헌](#-참고문헌)

---

## 🧭 프로젝트 개요

다양한 환경과 조건에서 생성된 딥페이크 콘텐츠를 정확하게 판별하는 AI 모델을 설계·학습합니다.  
데이터 전처리, 특징 추출, 모델 구조 설계 등 전 과정을 고려한 종합적인 접근이 요구되며,  
클래스 간 불균형 상황에서도 공정한 평가를 위해 **Macro F1-score** 를 핵심 목표 지표로 삼습니다.

```
Input (Image / Video Frame)
        │
        ▼
 ┌─────────────────┐
 │  Preprocessing  │  MTCNN Face Crop · Normalize · Augmentation
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  Backbone Model │  DDA (DINOv2 + LoRA) ← Best
 └────────┬────────┘  FreqNet / ViT-H/14 / NSG-VD / GenD / FSVFM
          │
          ▼
   Real / Fake (Binary Classification)
```

---

## 📊 데이터셋

### 데이터 구성

| 구분 | 수량 |
|:---:|---:|
| Total Train | 463,576 |
| Total Validation | 113,818 |
| Real | 170,000 |
| Fake | 407,394 |
| **Train : Val 비율** | **4 : 1** |
| **Real : Fake 비율** | **43 : 57** |

### 사용 데이터셋 목록

`StyleGAN_ffhq` · `faceapp` · `5k-fake` · `50k_fake` · `sfhq_pt2` · `sfhq_t2i` · `celebA_resize` · `ffhq_resize:v0` · `stable-diffusion` · `photoshop` · `deep_and_real` · `defaco` · `vggface2_30k`

### 생성 기법별 계열 분류

<details>
<summary><b>① 변형 및 합성 계열 (photoshop, defaco)</b></summary>

1990~2000년대에 등장한 **Face Morph** / **Face Swap** 방식입니다.  
얼굴의 기준 랜드마크 좌표를 삼각형으로 분할해 두 얼굴을 매칭·변형하는 방식으로, 경계·색상 톤 불일치 및 이질적인 노이즈 패턴으로 인해 상대적으로 탐지가 용이합니다.

</details>

<details>
<summary><b>② GAN 계열 (StyleGAN_ffhq, 50k_fake, sfhq)</b></summary>

2014년 GAN 등장 이후 **Autoencoder/GAN 기반**으로 얼굴 분포 자체를 학습하여 고해상도 가짜 얼굴 생성이 가능해졌습니다.  
그러나 **Mode Collapse** 문제와 명시적 likelihood 정의 불가로 인해 분포 Coverage 보장이 어렵다는 근본적 한계가 있습니다.

</details>

<details>
<summary><b>③ Transformer/Diffusion 혼합 계열 (sfhq_t2i, sfhq_pt2)</b></summary>

현재 가장 발전된 생성 AI 계열로, Convolution 블록 대신 **Transformer 블록**으로 구성되어 장거리 의존성 포착 및 이미지 전체의 복잡한 관계 파악에 탁월합니다.  
다만 노이즈 복원 과정이 완전하지 않아 특정 주파수 대역에 미묘한 생성 흔적이 남습니다.

</details>

---

## 🔧 전처리

| 기법 | 목적 |
|:---|:---|
| **RandomHorizontalFlip** | 좌우 반전으로 데이터 다양성 확보 (얼굴 레이블 불변) |
| **Normalize** | 채널별 평균·표준편차 정규화 → 수렴 안정화, 사전학습 가중치 호환 |
| **ColorJitter** | 밝기·대비·채도 랜덤 변화 → 조명·화이트밸런스 변화에 대한 Robustness |
| **ImageCompression** | 손실 압축 시뮬레이션 (Block noise, Ringing 아티팩트 생성) → 전송·저장 환경 일반화 |

> [!NOTE]
> `ImageCompression`의 품질 하한을 너무 낮게 설정하면 인식 불가능한 샘플이 생성됩니다. 하한 경계값 설정에 주의가 필요합니다.

---

## 🏗️ 모델 아키텍처

### FreqNet

<details>
<summary><b>세부 내용 보기</b></summary>

주파수 도메인에서 위조 흔적을 탐지하는 데 특화된 모델로, 세 가지 모듈을 통해 특징을 추출합니다.

```
Input Image
    │
    ▼
 [HFRI]  FFT → High-pass Filter → iFFT  (고주파 성분 강조)
    │
    ▼
 Residual Conv Block + [HFRF] + [FCL]  × N
    │   HFRF: 공간·채널 차원 FFT 고주파 필터링
    │   FCL : FFT → 진폭/위상 분리 → Conv → iFFT
    ▼
  GAP → FC → Binary Output
```

| 항목 | 값 |
|:---|:---|
| 입력 해상도 | 256×256 (MTCNN 1.3× crop) |
| Batch size | 128 |
| Optimizer | Adam (lr=1e-3, wd=1e-4) |
| Scheduler | CosineAnnealingLR |
| Augmentation | RandomHFlip · Normalize · ColorJitter · ImageCompression |

</details>

---

### DDA (Dual Data Alignment) ⭐ Best Model

<details>
<summary><b>세부 내용 보기</b></summary>

**DINOv2**를 백본으로 하여 **LoRA Fine-tuning**을 적용한 모델입니다.  
CLIP(고차원 의미 특화)과 DINOv2(저차원 시각 패턴 특화)를 비교 실험한 결과, 미세한 아티팩트 탐지에 뛰어난 DINOv2가 채택되었습니다.

**DDA 데이터 합성 파이프라인:**

```
Real Image (MS-COCO)
    │
    ├─► Stable Diffusion VAE 통과  →  생성 흔적 포함 이미지
    │         │
    │         ├─► 50% 확률로 JPEG 압축 적용  (고주파 정보 보정)
    │         │
    │         └─► Frequency Alignment  (실제 이미지와 주파수 정렬)
    │
    └─► Mixup (pixel-level 혼합, α ∈ [0.2, 0.8])
              │
              ▼
        최종 학습 데이터
```

| 항목 | 값 |
|:---|:---|
| LoRA Rank | 8 |
| Batch size | 16 (accumulation ×4 → effective 64) |
| Learning rate | 1e-4 |
| Train 전처리 | Random Crop 336×336 |
| Validation 전처리 | Center Crop 336×336 |
| Face Detector | MTCNN (41 frames) |

</details>

---

### NSG-VD

<details>
<summary><b>세부 내용 보기</b></summary>

물리적 보존 법칙 기반 **NSG(Normalized Spatiotemporal Gradient)** 로 비디오의 시공간적 역학을 모델링하고, **MMD(Maximum Mean Discrepancy)** 로 진위를 판별합니다.

```
Video Frames (8 frames uniform sampling)
    │
    ▼
NSG Feature Extraction
├─ 공간적 기울기: Diffusion model (256×256 unconditional ckpt)
└─ 시간적 미분: 밝기 불변성 가정 기반 픽셀 변위 근사
    │
    ▼
Swin Transformer
    │
    ▼
MMD-based Binary Classification (Threshold = 1.0)
```

| 항목 | 값 |
|:---|:---|
| 학습 데이터 | Kinetics-400 (10k) + Pika/SEINE 생성 (10k) |
| Batch size | 24 |
| Optimizer | Adam (lr=1e-4, wd=0.1) |
| Loss | MMD (Test Power 최대화) |
| 커널 파라미터 | σ=0.1, C=100, λ∈(0,1) |

</details>

---

### GenD

<details>
<summary><b>세부 내용 보기</b></summary>

**CLIP ViT-L/14** 를 백본으로 하여 전체 파라미터의 **0.03%** (Layer Normalization)만 미세 조정합니다.  
특징 공간을 초구(Hypersphere)로 제약하는 Metric Learning을 적용해 Shortcut Learning을 방지합니다.

| 항목 | 값 |
|:---|:---|
| 학습 데이터 | FaceForensics++ (FF++) |
| Batch size | 128 |
| Epochs | 20 |
| 샘플링 | Uniform (32 frames/video) |
| Face Detector | RetinaFace (landmark 정렬, 1.3× margin, 224×224) |
| Loss | CrossEntropy + Uniformity + Alignment |
| Scheduler | Linear Warm-up (1 epoch) + Cosine Decay |

</details>

---

### FSVFM

<details>
<summary><b>세부 내용 보기</b></summary>

대규모 **레이블 없는 실제 얼굴 데이터**로 사전학습된 비전 파운데이션 모델입니다.  
**MIM + Instance Discrimination** 기반 자가 지도 학습 프레임워크(3C 목표)를 사용합니다.

| 목표 | 설명 |
|:---:|:---|
| **Consistency** | CRFR-P 마스킹 → 영역 내 일관성 학습 |
| **Coherency** | 서로 다른 영역 간 유의미한 연결성 파악 |
| **Correspondence** | 마스킹 뷰 ↔ 원본 뷰 Local-to-Global 대응 |

| 항목 | 값 |
|:---|:---|
| Backbone | ViT + FS-Adapter (경량 병목 모듈) |
| Batch size | 64 |
| Base LR | 2.5e-4 |
| Epochs | 10 (ViT-B Scratch: 50) |
| Optimizer | AdamW + Cosine LR Schedule |

</details>

---

### ViT-H/14

<details>
<summary><b>세부 내용 보기</b></summary>

**LAION-2B** 사전학습 OpenCLIP **ViT-H/14** 와 **SRM(Spatial Rich Models) 필터**를 결합한 이중 입력 구조입니다.

```
Input Image
    ├─► RGB Branch         ──────────────────────────┐
    │                                                  ├─► 1×1 Conv (→ 3ch) ──► ViT-H/14
    └─► SRM Branch                                    │
        (Grayscale → High-pass Filter → Noise Map) ──┘
```

| 항목 | 값 |
|:---|:---|
| 입력 해상도 | 224×224 |
| Batch size | 64 |
| Optimizer | AdamW (lr=1e-3, wd=1e-4, β=[0.9, 0.999]) |
| Loss | CrossEntropyLoss |
| Scheduler | Cosine Scheduler |
| Augmentation | RandomHorizontalFlip · Normalize |

</details>

---

## 📈 실험 및 결과

### 실험 환경

| 항목 | 내용 |
|:---|:---|
| GPU | NVIDIA A100-SXM4-80GB |
| CPU | 12 Core |
| 실험 환경 | Google Colab |
| 하이퍼파라미터 관리 | Hydra |
| 실험 추적 | WandB |
| 학습 코드 | HuggingFace Accelerate (다중 CPU/GPU 대응) |

### 리더보드 성적

| Model | Dataset | Preprocess | **Score (Macro F1)** |
|:---:|:---:|:---:|:---:|
| **DDA** | DDA-COCO | MTCNN | **🥇 0.8194** |
| SPAI | DMID | DLIB · Masked Spectral Filtering | 0.5555 |
| FakeVLM | FakeClue | — | 0.5711 |
| FreqNet | Ours | MTCNN | 0.5253 |
| SBI | FaceForensics++ | STG · MG · Blending | 0.4694 |
| FSVFM | VGGFace2 | DLIB · RetinaFace | 0.4270 |
| GenD | FaceForensics++ | RetinaFace · Uniform Sampling | 0.4135 |
| SIDA | SID-Set | CLIP Processor · Resize & Padding | 0.3963 |
| AlignedForensics | MS-COCO · LSUN | Random Resized Crop | 0.3166 |
| ViT-H/14 | Ours | MTCNN | 0.3063 |

### Ablation Study

**FreqNet**

| Methods | Score (Macro F1) |
|:---|:---:|
| Baseline (BCE only) | 0.4472 |
| + RandomFlip · Normalize | 0.4944 |
| + F_beta · RandomFlip · Normalize | 0.4676 |
| + RandomFlip · Normalize · Jitter | 0.4949 |
| + Jitter · **ImageCompression** ✅ | **0.5253** |

**DDA**

| Methods | Score (Macro F1) |
|:---|:---:|
| Center Crop (Baseline) | 0.8168 |
| RetinaFace crop (11 frames) | 0.8149 |
| **MTCNN crop (41 frames)** ✅ | **0.8194** |
| DDA + NSG-VD (video branch 분리) | 0.7895 |

> [!NOTE]
> **ImageCompression** 증강이 FreqNet 성능을 크게 향상시킨 이유는, DDA 논문에서 언급된 것과 같이 압축이 생성 모델의 풍부한 아티팩트 정보를 희석시켜 모델의 일반화 성능을 높였기 때문으로 분석됩니다.

> [!NOTE]
> **NSG-VD 분리 전략**이 DDA 단독보다 낮은 성능을 보인 이유는, 이미지 모델(DDA)이 이미 동영상에서 추출된 프레임을 학습 데이터로 포함하고 있어 비디오 전용 모델 대비 불리한 점이 없었으며, NSG-VD가 GAN/전통 위조보다 생성형 모델 아티팩트에 특화되어 있기 때문입니다.

---

## 📚 참고문헌

| Model | Paper | Code |
|:---:|:---:|:---:|
| FreqNet | [arXiv:2403.07240](https://arxiv.org/abs/2403.07240) | [GitHub](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection) |
| DDA | [arXiv:2505.14359](https://arxiv.org/abs/2505.14359) | [GitHub](https://github.com/roy-ch/Dual-Data-Alignment) |
| NSG-VD | [OpenReview](https://openreview.net/forum?id=HiBoJLCyEo) | [GitHub](https://github.com/ZSHsh98/NSG-VD) |
| GenD | [arXiv:2508.06248](https://arxiv.org/abs/2508.06248) | [GitHub](https://github.com/yermandy/GenD) |
| FSVFM | [arXiv:2510.10663](https://arxiv.org/abs/2510.10663) | [GitHub](https://github.com/wolo-wolo/FSFM-CVPR25/tree/FSVFM-extension) |
| AlignedForensics | [arXiv:2410.11835](https://arxiv.org/abs/2410.11835) | [GitHub](https://github.com/AniSundar18/AlignedForensics) |
| SPAI | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html) | [GitHub](https://github.com/mever-team/spai) |
| SBI | [arXiv:2204.08376](https://arxiv.org/abs/2204.08376) | [GitHub](https://github.com/mapooon/SelfBlendedImages) |
| SIDA | [arXiv:2412.04292](https://arxiv.org/abs/2412.04292) | [GitHub](https://github.com/hzlsaber/SIDA) |
| FakeVLM | [arXiv:2503.14905](https://arxiv.org/abs/2503.14905) | [GitHub](https://github.com/opendatalab/FakeVLM) |
| AIDE | [arXiv:2502.13138](https://arxiv.org/abs/2502.13138) | [GitHub](https://github.com/WecoAI/aideml) |

---

<div align="center">
  <sub>AI Factory Deepfake Detection Challenge</sub>
</div>

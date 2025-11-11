"""
전처리 모듈 테스트 스크립트

사용법:
    python test_preprocessing.py --image_path path/to/image.jpg
    python test_preprocessing.py --data_root path/to/dataset
"""

import argparse
from pathlib import Path
import sys

# src를 Python path에 추가
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from PIL import Image
import dlib
import torch
from torchvision import transforms

from preprocessing import detect_and_crop_face
from preprocessing.datasets import FaceDetectionDataset, StandardDataset


def test_face_detection(image_path):
    """얼굴 검출 테스트"""
    print("="*50)
    print("1. 얼굴 검출 테스트")
    print("="*50)
    
    # 경로 확인 및 처리
    path = Path(image_path)
    
    # dlib 탐지기 초기화 (한 번만)
    detector = dlib.get_frontal_face_detector()
    print("dlib 탐지기 초기화 완료")
    
    # 디렉토리인 경우 모든 이미지 파일 처리
    if path.is_dir():
        print(f"\n디렉토리 모드: {path}")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in path.iterdir() if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"✗ 디렉토리에 이미지 파일이 없습니다: {image_path}")
            return False
        
        print(f"총 {len(image_files)}개의 이미지 발견\n")
        
        # 출력 디렉토리 생성
        output_dir = path / "face_detected"
        output_dir.mkdir(exist_ok=True)
        print(f"결과 저장 디렉토리: {output_dir}\n")
        
        success_count = 0
        fail_count = 0
        
        # 모든 이미지 처리
        for i, img_file in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] 처리 중: {img_file.name}")
            
            try:
                # 이미지 로드
                image = Image.open(img_file)
                print(f"  원본 크기: {image.size}")
                
                # 얼굴 검출 및 크롭
                face_img = detect_and_crop_face(
                    image,
                    detector,
                    target_size=(256, 256),
                    resize_for_detection=640,
                    scale_factor=1.5
                )
                
                if face_img is not None:
                    print(f"  ✓ 얼굴 검출 성공! 크롭 크기: {face_img.size}")
                    # 결과 저장
                    output_path = output_dir / f"{img_file.stem}_face.jpg"
                    face_img.save(output_path)
                    print(f"  ✓ 저장 완료: {output_path.name}")
                    success_count += 1
                else:
                    print(f"  ✗ 얼굴 검출 실패")
                    fail_count += 1
                    
            except Exception as e:
                print(f"  ✗ 처리 실패: {e}")
                fail_count += 1
            
            print()  # 빈 줄
        
        # 최종 결과
        print("="*50)
        print(f"처리 완료: 성공 {success_count}개 / 실패 {fail_count}개")
        print(f"결과 위치: {output_dir}")
        print("="*50)
        
        return success_count > 0
    
    # 단일 파일 처리
    else:
        print(f"단일 파일 모드: {path.name}")
        
        # 이미지 로드
        try:
            image = Image.open(path)
            print(f"원본 이미지 크기: {image.size}")
        except Exception as e:
            print(f"✗ 이미지 로드 실패: {e}")
            return False
        
        # 얼굴 검출 및 크롭
        face_img = detect_and_crop_face(
            image,
            detector,
            target_size=(256, 256),
            resize_for_detection=640,
            scale_factor=1.5
        )
        
        if face_img is not None:
            print(f"✓ 얼굴 검출 성공! 크롭된 이미지 크기: {face_img.size}")
            # 결과 저장
            output_path = path.parent / f"face_detected_{path.stem}.jpg"
            face_img.save(output_path)
            print(f"✓ 결과 저장: {output_path}")
        else:
            print("✗ 얼굴 검출 실패")
        
        return face_img is not None


def test_dataset(data_root):
    """Dataset 테스트"""
    print("\n" + "="*50)
    print("2. Dataset 테스트")
    print("="*50)
    
    # Transform 정의
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    
    # FaceDetectionDataset 테스트
    print("\n[FaceDetectionDataset 테스트]")
    try:
        face_dataset = FaceDetectionDataset(
            root=data_root,
            transform=transform,
            target_size=(256, 256),
            return_original_on_fail=True
        )
        print(f"✓ 데이터셋 로드 성공: {len(face_dataset)} 샘플")
        
        # 첫 번째 샘플 테스트
        if len(face_dataset) > 0:
            img, label = face_dataset[0]
            print(f"  샘플 shape: {img.shape}, 라벨: {label}")
            print(f"  클래스: {face_dataset.classes}")
    except Exception as e:
        print(f"✗ FaceDetectionDataset 오류: {e}")
    
    # StandardDataset 테스트
    print("\n[StandardDataset 테스트]")
    try:
        standard_dataset = StandardDataset(
            root=data_root,
            transform=transform
        )
        print(f"✓ 데이터셋 로드 성공: {len(standard_dataset)} 샘플")
        
        # 첫 번째 샘플 테스트
        if len(standard_dataset) > 0:
            img, label = standard_dataset[0]
            print(f"  샘플 shape: {img.shape}, 라벨: {label}")
            print(f"  클래스: {standard_dataset.classes}")
    except Exception as e:
        print(f"✗ StandardDataset 오류: {e}")


def test_dataloader(data_root):
    """DataLoader 테스트"""
    print("\n" + "="*50)
    print("3. DataLoader 테스트")
    print("="*50)
    
    from torch.utils.data import DataLoader
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    try:
        dataset = FaceDetectionDataset(
            root=data_root,
            transform=transform,
            target_size=(256, 256)
        )
        
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0  # 테스트는 단일 프로세스
        )
        
        print(f"DataLoader 생성 완료: {len(loader)} 배치")
        
        # 첫 배치 로드 테스트
        for batch_idx, (images, labels) in enumerate(loader):
            print(f"  배치 {batch_idx}: images shape={images.shape}, labels={labels}")
            if batch_idx >= 2:  # 처음 3개 배치만
                break
        
        print("✓ DataLoader 테스트 성공")
        
    except Exception as e:
        print(f"✗ DataLoader 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='전처리 모듈 테스트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 단일 이미지 또는 폴더에서 얼굴 검출 테스트
  python test_preprocessing.py --image_path path/to/image.jpg
  python test_preprocessing.py --image_path path/to/image_folder/
  
  # 데이터셋 테스트
  python test_preprocessing.py --data_root path/to/dataset/
  python test_preprocessing.py --data_root path/to/dataset/train/
  
  지원하는 디렉토리 구조:
    1) split 폴더 포함 (자동으로 train 폴더 사용):
       dataset/              <- --data_root 경로
       ├── train/
       │   ├── 0/
       │   └── 1/
       └── val/
           ├── 0/
           └── 1/
    
    2) 직접 클래스 폴더:
       dataset/              <- --data_root 경로
       ├── 0/
       └── 1/
        """
    )
    parser.add_argument('--image_path', type=str, 
                       help='테스트할 이미지 파일 또는 이미지가 있는 폴더 경로')
    parser.add_argument('--data_root', type=str, 
                       help='데이터셋 루트 경로 (클래스 폴더들이 있는 상위 디렉토리)')
    args = parser.parse_args()
    
    if not args.image_path and not args.data_root:
        print("오류: --image_path 또는 --data_root 중 하나를 지정해주세요")
        parser.print_help()
        return
    
    # 얼굴 검출 테스트
    if args.image_path:
        if Path(args.image_path).exists():
            test_face_detection(args.image_path)
        else:
            print(f"오류: 이미지를 찾을 수 없습니다: {args.image_path}")
    
    # Dataset 테스트
    if args.data_root:
        data_root_path = Path(args.data_root)
        if data_root_path.exists():
            # 실제 테스트할 경로 결정
            test_path = None
            
            # 경로 구조 확인
            subdirs = [d for d in data_root_path.iterdir() if d.is_dir()]
            subdir_names = [d.name.lower() for d in subdirs]
            
            # 1. 직접 클래스 폴더가 있는 경우 (0, 1, fake, real 등)
            if any(name in ['0', '1', 'fake', 'real', 'true'] for name in subdir_names):
                print(f"\n✓ 클래스 폴더 직접 발견: {[d.name for d in subdirs]}")
                test_path = data_root_path
            
            # 2. train/val 같은 split 폴더가 있는 경우
            elif 'train' in subdir_names:
                train_path = data_root_path / 'train'
                print(f"\n✓ train 폴더 발견: {train_path}")
                train_subdirs = [d for d in train_path.iterdir() if d.is_dir()]
                print(f"  클래스 폴더: {[d.name for d in train_subdirs]}")
                
                # 클래스 폴더가 없으면 구조 상세 출력
                if not train_subdirs:
                    print("\n⚠ train 폴더에 하위 폴더가 없습니다!")
                    print("  폴더 내용 확인:")
                    all_items = list(train_path.iterdir())
                    if not all_items:
                        print("    (비어있음)")
                    else:
                        for item in all_items[:10]:  # 최대 10개만
                            item_type = "📁" if item.is_dir() else "📄"
                            print(f"    {item_type} {item.name}")
                        if len(all_items) > 10:
                            print(f"    ... 외 {len(all_items) - 10}개")
                    return
                
                test_path = train_path
            
            elif 'val' in subdir_names or 'validation' in subdir_names:
                val_path = data_root_path / ('val' if 'val' in subdir_names else 'validation')
                print(f"\n✓ validation 폴더 발견: {val_path}")
                val_subdirs = [d for d in val_path.iterdir() if d.is_dir()]
                print(f"  클래스 폴더: {[d.name for d in val_subdirs]}")
                
                # 클래스 폴더가 없으면 구조 상세 출력
                if not val_subdirs:
                    print("\n⚠ validation 폴더에 하위 폴더가 없습니다!")
                    print("  폴더 내용 확인:")
                    all_items = list(val_path.iterdir())
                    if not all_items:
                        print("    (비어있음)")
                    else:
                        for item in all_items[:10]:  # 최대 10개만
                            item_type = "📁" if item.is_dir() else "📄"
                            print(f"    {item_type} {item.name}")
                        if len(all_items) > 10:
                            print(f"    ... 외 {len(all_items) - 10}개")
                    return
                
                test_path = val_path
            
            else:
                print(f"\n⚠ 경고: {args.data_root} 에서 적절한 데이터셋 구조를 찾을 수 없습니다.")
                print(f"발견된 하위 폴더: {[d.name for d in subdirs]}")
                print("\n지원하는 구조:")
                print("  1) 직접 클래스 폴더:")
                print("     dataset/")
                print("     ├── 0/")
                print("     └── 1/")
                print("\n  2) split 폴더 포함:")
                print("     dataset/")
                print("     ├── train/")
                print("     │   ├── 0/")
                print("     │   └── 1/")
                print("     └── val/")
                print("         ├── 0/")
                print("         └── 1/")
                print("\n힌트: train 또는 val 폴더를 직접 지정할 수도 있습니다:")
                print(f"  python test_preprocessing.py --data_root \"{data_root_path / 'train'}\"")
                return
            
            if test_path:
                test_dataset(str(test_path))
                test_dataloader(str(test_path))
        else:
            print(f"오류: 데이터셋 경로를 찾을 수 없습니다: {args.data_root}")
    
    print("\n" + "="*50)
    print("테스트 완료")
    print("="*50)


if __name__ == '__main__':
    main()


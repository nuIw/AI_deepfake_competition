"""
데이터셋 라벨 폴더 정리 스크립트

데이터셋 구조를 정규화합니다:
1. Split 폴더 이름 정규화: validation → val
2. 라벨 폴더 정규화:
   - fake 폴더의 이미지들을 1 폴더로 이동
   - true 폴더의 이미지들을 0 폴더로 이동

사용법:
    python src/organize_labels.py --data_path /path/to/dataset
    python src/organize_labels.py --data_path /path/to/dataset --dry-run  # 시뮬레이션만
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def organize_labels(data_path, dry_run=False):
    """
    데이터셋의 구조를 정규화합니다.
    
    1단계: Split 폴더 이름 정규화 (validation → val)
    2단계: 라벨 폴더 정규화 (fake→1, true→0)
    
    Args:
        data_path: 데이터셋 경로 (train/val/test가 있는 최상위 디렉토리)
        dry_run: True면 실제로 이동하지 않고 시뮬레이션만 수행
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"데이터셋 경로를 찾을 수 없습니다: {data_path}")
    
    # 1단계: Split 폴더 이름 정규화 및 병합 (validation → val)
    print("\n" + "="*50)
    print("1단계: Split 폴더 이름 정규화 및 병합")
    print("="*50)
    
    # split 이름 매핑: 다양한 이름을 표준 이름으로 변경
    split_name_mapping = {
        'validation': 'val',
        'Validation': 'val',
        'VALIDATION': 'val',
        'valid': 'val',
        'Valid': 'val',
        'VALID': 'val',
        'dev': 'val',
        'Dev': 'val',
        'DEV': 'val',
        'development': 'val',
    }
    
    # data_path의 직접 하위 디렉토리 중 split 폴더 찾기
    for item in data_path.iterdir():
        if not item.is_dir():
            continue
        
        item_name = item.name
        # 이미 표준 이름이거나 매핑에 없으면 건너뛰기
        if item_name in ['train', 'val', 'test']:
            continue
        
        # 매핑에 있는 경우 처리
        if item_name in split_name_mapping:
            new_name = split_name_mapping[item_name]
            new_path = data_path / new_name
            
            if new_path.exists():
                # 이미 val 폴더가 존재하는 경우: validation 폴더의 내용을 val로 병합
                print(f"  📦 [{item_name}] → [{new_name}]: 이미 [{new_name}] 폴더가 존재합니다. 내용을 병합합니다.")
                
                if dry_run:
                    # validation 내부의 파일 개수 확인
                    total_files = sum(1 for _ in item.rglob('*') if _.is_file())
                    print(f"     [DRY RUN] {total_files}개 파일/폴더를 [{new_name}]로 병합할 예정")
                else:
                    try:
                        # validation 폴더 내의 모든 내용을 val로 이동/병합
                        merged_count = 0
                        for content in item.iterdir():
                            content_name = content.name
                            target_content = new_path / content_name
                            
                            if content.is_dir():
                                # 디렉토리인 경우
                                if target_content.exists():
                                    # 같은 이름의 폴더가 이미 존재: 내용을 병합
                                    print(f"       📂 [{content_name}] 폴더가 이미 존재합니다. 내용을 병합합니다.")
                                    
                                    # validation/content_name 내의 모든 파일/폴더를 val/content_name으로 이동
                                    sub_merged = 0
                                    for sub_content in content.iterdir():
                                        sub_content_name = sub_content.name
                                        sub_target = target_content / sub_content_name
                                        
                                        # 중복 이름 처리
                                        counter = 1
                                        while sub_target.exists():
                                            if sub_content.is_file():
                                                stem = sub_content.stem
                                                suffix = sub_content.suffix
                                                sub_target = target_content / f"{stem}_{counter}{suffix}"
                                            else:
                                                sub_target = target_content / f"{sub_content_name}_{counter}"
                                            counter += 1
                                        
                                        sub_content.rename(sub_target)
                                        sub_merged += 1
                                    
                                    # validation/content_name 폴더가 비어있으면 삭제
                                    try:
                                        if not any(content.iterdir()):
                                            content.rmdir()
                                    except:
                                        pass
                                    
                                    merged_count += sub_merged
                                else:
                                    # 같은 이름의 폴더가 없음: 그냥 이동
                                    content.rename(target_content)
                                    merged_count += 1
                            else:
                                # 파일인 경우: 중복 이름 처리 후 이동
                                counter = 1
                                while target_content.exists():
                                    stem = content.stem
                                    suffix = content.suffix
                                    target_content = new_path / f"{stem}_{counter}{suffix}"
                                    counter += 1
                                
                                content.rename(target_content)
                                merged_count += 1
                        
                        print(f"     ✓ {merged_count}개 항목을 [{new_name}]로 병합 완료")
                        
                        # validation 폴더가 비어있으면 삭제
                        try:
                            if not any(item.iterdir()):
                                item.rmdir()
                                print(f"     ✓ 빈 [{item_name}] 폴더 삭제 완료")
                        except Exception as e:
                            print(f"     ⚠ [{item_name}] 폴더 삭제 실패: {e}")
                            
                    except Exception as e:
                        print(f"     ⚠ [{item_name}] → [{new_name}] 병합 실패: {e}")
            else:
                # val 폴더가 없으면 이름만 변경
                if dry_run:
                    print(f"  [DRY RUN] [{item_name}] → [{new_name}] 폴더 이름 변경 예정")
                else:
                    try:
                        item.rename(new_path)
                        print(f"  ✓ [{item_name}] → [{new_name}] 폴더 이름 변경 완료")
                    except Exception as e:
                        print(f"  ⚠ [{item_name}] → [{new_name}] 폴더 이름 변경 실패: {e}")
    
    # fake를 나타내는 폴더명들 (소문자로 변환하여 비교)
    fake_labels = {'fake', '1', 'fakes', 'FAKE', 'Fake'}
    # real을 나타내는 폴더명들
    real_labels = {'true', '0', 'real', 'reals', 'TRUE', 'True', 'REAL', 'Real'}
    
    # train, val, test 디렉토리 처리
    splits = ['train', 'val', 'test']
    
    total_moved = 0
    total_skipped = 0
    
    # 2단계: 라벨 폴더 정리 (fake→1, true→0)
    print("\n" + "="*50)
    print("2단계: 라벨 폴더 정리 (fake→1, true→0)")
    print("="*50)
    
    for split in splits:
        split_path = data_path / split
        
        if not split_path.exists():
            print(f"⚠ [{split}] 디렉토리가 없습니다. 건너뜁니다.")
            continue
        
        print(f"\n📁 [{split}] 처리 중...")
        
        # 각 라벨 폴더 탐색
        label_folders = [d for d in split_path.iterdir() if d.is_dir()]
        
        if not label_folders:
            print(f"  ⚠ 라벨 폴더를 찾을 수 없습니다.")
            continue
        
        # 목적지 폴더 생성 (0, 1)
        target_real_dir = split_path / '0'
        target_fake_dir = split_path / '1'
        
        if not dry_run:
            target_real_dir.mkdir(exist_ok=True)
            target_fake_dir.mkdir(exist_ok=True)
        
        for folder in label_folders:
            folder_name = folder.name
            folder_name_lower = folder_name.lower()
            
            # 이미 정규화된 폴더는 건너뛰기
            if folder_name in {'0', '1'}:
                print(f"  ✓ [{folder_name}] 이미 정규화되어 있습니다. 건너뜁니다.")
                continue
            
            # 라벨 결정
            if folder_name_lower in {f.lower() for f in fake_labels}:
                target_dir = target_fake_dir
                label_type = 'fake (1)'
            elif folder_name_lower in {f.lower() for f in real_labels}:
                target_dir = target_real_dir
                label_type = 'real (0)'
            else:
                print(f"  ⚠ 알 수 없는 폴더명: [{folder_name}]. 건너뜁니다.")
                total_skipped += 1
                continue
            
            # 폴더 내 이미지 파일 찾기
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG'}
            image_files = [f for f in folder.iterdir() 
                          if f.is_file() and f.suffix.lower() in image_extensions]
            
            if not image_files:
                print(f"  ⚠ [{folder_name}] 이미지 파일이 없습니다.")
                continue
            
            print(f"  📦 [{folder_name}] → {label_type} ({len(image_files)}개 파일)")
            
            if dry_run:
                print(f"     [DRY RUN] {len(image_files)}개 파일을 {target_dir.name}/로 이동할 예정")
                total_moved += len(image_files)
            else:
                # 파일 이동
                moved_count = 0
                for img_file in tqdm(image_files, desc=f"    이동 중", leave=False):
                    try:
                        target_path = target_dir / img_file.name
                        # 중복 파일명 처리
                        counter = 1
                        while target_path.exists():
                            stem = img_file.stem
                            suffix = img_file.suffix
                            target_path = target_dir / f"{stem}_{counter}{suffix}"
                            counter += 1
                        
                        shutil.move(str(img_file), str(target_path))
                        moved_count += 1
                    except Exception as e:
                        print(f"     ⚠ 오류: {img_file.name} 이동 실패 - {e}")
                
                total_moved += moved_count
                
                # 원본 폴더가 비어있으면 삭제
                try:
                    remaining_files = list(folder.iterdir())
                    if not remaining_files:
                        folder.rmdir()
                        print(f"     ✓ 빈 폴더 삭제: {folder_name}")
                except Exception as e:
                    print(f"     ⚠ 폴더 삭제 실패: {e}")
    
    print(f"\n{'='*50}")
    print("최종 결과")
    print('='*50)
    if dry_run:
        print(f"📊 [DRY RUN] 총 {total_moved}개 파일 이동 예정")
        print(f"   {total_skipped}개 폴더 건너뜀")
    else:
        print(f"✅ 완료! 총 {total_moved}개 파일 이동")
        print(f"   {total_skipped}개 폴더 건너뜀")
    
    return total_moved, total_skipped


def main():
    parser = argparse.ArgumentParser(
        description='데이터셋 라벨 폴더 정리 (fake→1, true→0)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 실제 이동
  python src/organize_labels.py --data_path /path/to/dataset
  
  # 시뮬레이션만 (실제로 이동하지 않음)
  python src/organize_labels.py --data_path /path/to/dataset --dry-run
  
  # train/val/test가 직접 있는 경우
  python src/organize_labels.py --data_path /path/to/train
        """
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='데이터셋 경로 (train/val/test가 있는 최상위 디렉토리 또는 train/val/test 중 하나)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='실제로 이동하지 않고 시뮬레이션만 수행'
    )
    
    args = parser.parse_args()
    
    print("="*50)
    print("데이터셋 라벨 폴더 정리")
    print("="*50)
    print(f"데이터셋 경로: {args.data_path}")
    print(f"모드: {'DRY RUN (시뮬레이션)' if args.dry_run else '실제 이동'}")
    print("="*50)
    
    try:
        organize_labels(args.data_path, dry_run=args.dry_run)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


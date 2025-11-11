"""
데이터셋 구조 확인 스크립트

사용법:
    python check_dataset_structure.py path/to/dataset
"""

import sys
from pathlib import Path


def print_tree(path, prefix="", max_depth=3, current_depth=0, max_files=5):
    """디렉토리 구조를 트리 형태로 출력"""
    if current_depth >= max_depth:
        return
    
    path = Path(path)
    if not path.exists():
        print(f"경로를 찾을 수 없습니다: {path}")
        return
    
    if not path.is_dir():
        print(f"디렉토리가 아닙니다: {path}")
        return
    
    try:
        items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        dirs = [item for item in items if item.is_dir()]
        files = [item for item in items if item.is_file()]
        
        # 디렉토리 출력
        for i, item in enumerate(dirs):
            is_last = (i == len(dirs) - 1) and len(files) == 0
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}[DIR] {item.name}/")
            
            extension = "    " if is_last else "│   "
            print_tree(item, prefix + extension, max_depth, current_depth + 1, max_files)
        
        # 파일 출력 (개수 제한)
        file_count = len(files)
        display_files = files[:max_files]
        
        for i, item in enumerate(display_files):
            is_last = (i == len(display_files) - 1) and (file_count <= max_files)
            connector = "└── " if is_last else "├── "
            
            # 파일 확장자에 따라 표시
            ext = item.suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                icon = "[IMG]"
            elif ext in ['.mp4', '.avi', '.mov']:
                icon = "[VID]"
            else:
                icon = "[FILE]"
            
            print(f"{prefix}{connector}{icon} {item.name}")
        
        # 더 많은 파일이 있으면 표시
        if file_count > max_files:
            connector = "└── " if True else "├── "
            print(f"{prefix}{connector}... 외 {file_count - max_files}개 파일")
    
    except PermissionError:
        print(f"{prefix}(접근 권한 없음)")


def analyze_structure(path):
    """데이터셋 구조 분석"""
    path = Path(path)
    print("="*60)
    print("데이터셋 구조 분석")
    print("="*60)
    print(f"경로: {path}")
    print()
    
    if not path.exists():
        print(f"❌ 경로를 찾을 수 없습니다: {path}")
        return
    
    # 구조 출력
    print("디렉토리 구조:")
    print(f"{path.name}/")
    print_tree(path, "", max_depth=4, max_files=3)
    
    print()
    print("="*60)
    print("분석 결과")
    print("="*60)
    
    # 통계
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    files = [f for f in path.iterdir() if f.is_file()]
    
    print(f"직접 하위 폴더: {len(subdirs)}개")
    if subdirs:
        print(f"  폴더 이름: {[d.name for d in subdirs]}")
    
    print(f"직접 하위 파일: {len(files)}개")
    
    # train/val 폴더 확인
    subdir_names = [d.name.lower() for d in subdirs]
    
    if 'train' in subdir_names or 'val' in subdir_names:
        print("\n[OK] split 폴더 구조 감지 (train/val)")
        
        for split_name in ['train', 'val', 'test']:
            split_path = path / split_name
            if split_path.exists():
                class_dirs = [d for d in split_path.iterdir() if d.is_dir()]
                print(f"\n  [{split_name}]")
                print(f"    클래스 폴더: {len(class_dirs)}개")
                if class_dirs:
                    for cls_dir in class_dirs:
                        img_count = len([f for f in cls_dir.iterdir() if f.is_file()])
                        print(f"      {cls_dir.name}: {img_count}개 파일")
    
    elif any(name in ['0', '1', 'fake', 'real', 'true'] for name in subdir_names):
        print("\n[OK] 직접 클래스 폴더 구조 감지")
        for cls_dir in subdirs:
            img_count = len([f for f in cls_dir.iterdir() if f.is_file()])
            print(f"  {cls_dir.name}: {img_count}개 파일")
    
    else:
        print("\n[WARNING] 표준 데이터셋 구조가 아닙니다")
        print("\n권장 구조:")
        print("  1) split 폴더:")
        print("     dataset/")
        print("     ├── train/")
        print("     │   ├── 0/")
        print("     │   └── 1/")
        print("     └── val/")
        print("         ├── 0/")
        print("         └── 1/")
        print("\n  2) 직접 클래스 폴더:")
        print("     dataset/")
        print("     ├── 0/")
        print("     └── 1/")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("사용법: python check_dataset_structure.py <데이터셋_경로>")
        print()
        print("예시:")
        print("  python check_dataset_structure.py C:\\Users\\user\\Downloads\\samples")
        print("  python check_dataset_structure.py ./data")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    analyze_structure(dataset_path)


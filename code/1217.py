import glob
import natsort
import os
from collections import defaultdict, Counter

# 파일 경로 설정 (visual_30 폴더만 탐색)
visual = natsort.natsorted(glob.glob('./images_auto/visual_window30/***/**/*.jpg'))

# 클래스별 파일명 정리
file_dict = defaultdict(list)

# 파일명에서 클래스명과 숫자 추출
for file_path in visual:
    filename = os.path.basename(file_path)
    class_name = filename.split('_')[0] + '_' + filename.split('_')[1]  # 클래스명 추출
    try:
        file_number = int(filename.split('_')[-1].split('.')[0])  # 숫자 추출
        file_dict[class_name].append((file_number, file_path))  # (숫자, 파일 경로) 저장
    except (ValueError, IndexError):
        print(f"Warning: Failed to parse {filename}")  # 오류 출력
        continue

# 중복된 숫자 확인 및 출력
for class_name, files in file_dict.items():
    number_counter = Counter([num for num, _ in files])  # 숫자별 개수 카운트
    duplicates = [num for num, count in number_counter.items() if count > 1]  # 중복 숫자 찾기

    print(f"\nClass: {class_name}")
    print(f"  Total files: {len(files)}")

    if duplicates:
        print(f"  Duplicated numbers: {duplicates}")
        print("  Duplicate file paths:")
        for num in duplicates:
            duplicate_files = [path for n, path in files if n == num]
            for file in duplicate_files:
                print(f"    {file}")
    else:
        print("  All numbers from 0 to 32 are unique.")

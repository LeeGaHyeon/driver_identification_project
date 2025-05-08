import os

# 특정 경로에 있는 이미지 파일들의 확장자를 정의합니다.
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

def count_images_in_directory(directory):
    """주어진 디렉토리 내에서 이미지 파일 개수를 셉니다."""
    return sum(1 for file in os.listdir(directory) if file.lower().endswith(IMAGE_EXTENSIONS))

def count_images(base_path):
    # A, B, C 경로마다 이미지 개수를 저장할 딕셔너리
    abc_counts = {'A': 0, 'B': 0, 'C': 0}
    # 1, 2, 3, 4 경로마다 이미지 개수를 저장할 딕셔너리
    number_counts = {'1': 0, '2': 0, '3': 0, '4': 0}
    
    total_images = 0
    
    # base_path 내의 이름 폴더 탐색 (사람 이름 폴더 무시)
    for person_folder in os.listdir(base_path):
        person_path = os.path.join(base_path, person_folder)
        
        # 만약 디렉토리가 아니라면 건너뜀
        if not os.path.isdir(person_path):
            continue
        
        # A, B, C 폴더 탐색
        for abc_folder in ['A', 'B', 'C']:
            abc_path = os.path.join(person_path, abc_folder)
            
            # bump와 corner 폴더를 탐색
            for main_folder in ['bump', 'corner']:
                main_path = os.path.join(abc_path, main_folder)
                
                # 1, 2, 3, 4 폴더만 탐색
                for number_folder in ['1', '2', '3', '4']:
                    number_path = os.path.join(main_path, number_folder)
                    
                    # 경로가 존재하는지 확인
                    if not os.path.exists(number_path):
                        print(f"경로 없음: {number_path}")
                        continue
                    
                    # 이미지 파일 개수 세기
                    image_count = count_images_in_directory(number_path)
                    print(f"{number_path} : {image_count}")
                    
                    # 각 폴더의 개수를 누적
                    abc_counts[abc_folder] += image_count
                    number_counts[number_folder] += image_count
                    
                    # 전체 이미지 개수도 누적
                    total_images += image_count
    
    return abc_counts, number_counts, total_images

# 기본 경로를 정의
base_directory = "images/visual_window30/"

# 결과 출력
abc_image_counts, number_image_counts, total_image_count = count_images(base_directory)

print("A, B, C 경로 별 이미지 개수:", abc_image_counts)
print("1, 2, 3, 4 경로 별 이미지 개수:", number_image_counts)
print("전체 이미지 개수 총합:", total_image_count)




# import os

# # 이미지 파일 확장자를 정의합니다.
# IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

# def count_images_in_directory(directory):
#     """주어진 디렉토리 내에서 이미지 파일 개수를 셉니다."""
#     return sum(1 for file in os.listdir(directory) if file.lower().endswith(IMAGE_EXTENSIONS))

# def find_and_count_images_in_lowest_subdirectories(base_path):
#     """모든 하위 폴더를 탐색하고, 하위 폴더 내의 이미지 파일 개수를 계산합니다."""
#     total_images = 0
    
#     # os.walk를 사용하여 하위 디렉토리를 순차적으로 탐색
#     for root, dirs, files in os.walk(base_path):
#         if not dirs:  # 하위 폴더가 없는 경우
#             # 가장 하위 폴더에 있는 이미지 파일 개수를 셉니다.
#             image_count = count_images_in_directory(root)
#             #print(f"Folder: {root}, Image Count: {image_count}")
#             total_images += image_count
    
#     return total_images

# # 기본 경로를 정의
# base_directory = "images/optical_window30/"

# # 결과 출력
# total_image_count = find_and_count_images_in_lowest_subdirectories(base_directory)

# print(f"전체 하위 폴더에서 이미지 개수 총합: {total_image_count}")

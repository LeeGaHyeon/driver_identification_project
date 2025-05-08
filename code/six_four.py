import glob
import os
from natsort import natsorted

visual_paths = natsorted(glob.glob('./images/optical_window30/*/*/*/*/*.jpg'))
 
# classes = {'anseonyeong':0, 'baekseungdo':1, 'cheonaeji':2, 'jojeongduk':3, 'kimminju':4, 'leeeunseo':5, 'seosanghyeok':6}
classes = {'baekseungdo':1}
courses = ['A', 'B', 'C']
event_classes = {'bump': 0, 'corner': 1}
counts = ['1', '2', '3', '4', '5', '6']

# 경로 내에서 파일명 변경 및 삭제 작업 수행
for path in visual_paths:
    # 현재 경로를 분리하여 필요한 부분을 추출합니다
    parts = path.split('/')

    # 해당 디렉토리에 있는 파일들을 가져옵니다
    current_dir = os.path.dirname(path)
    existing_files = natsorted(glob.glob(os.path.join(current_dir, '*.jpg')))

    if len(existing_files) == 6:
        # 6개의 파일이 있는 경우 첫 두 파일을 삭제합니다
        os.remove(existing_files[0])
        os.remove(existing_files[1])

        # 남은 파일들을 1, 2, 3, 4로 리네임합니다
        for idx, file_path in enumerate(existing_files[2:], start=1):
            new_filename = f"{idx}.jpg"
            new_path = os.path.join(current_dir, new_filename)

            os.rename(file_path, new_path)

    # 4개 파일이면 그대로 유지 (기존 파일들은 이미 정렬된 상태)

print("파일 경로 내 작업이 완료되었습니다.")

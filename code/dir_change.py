import glob
import os
import shutil
from natsort import natsorted

visual_paths = natsorted(glob.glob('./images/optical_window30_new/*/*/*/*/*.jpg'))
 
# 특정 클래스만 선택하여 작업하기 위해 클래스 목록 정의
classes = {'choimingi':3, 'jeongyubin':4, 'leegahyeon':8, 'leegihun':9, 'leejaeho':10, 'leekanghyuk':11, 'leeyunguel':12, 'simboseok':14}
courses = ['A', 'B', 'C']
event_classes = {'bump': 0, 'corner': 1}
counts = ['1', '2', '3', '4']

# 지정된 클래스에 해당하는 파일만 새 경로로 이동
for path in visual_paths:
    # 현재 경로를 분리하여 필요한 부분을 추출합니다
    parts = path.split('/')

    class_name = parts[3]
    event_class = parts[4]
    course = parts[5]
    count = parts[6]
    filename = parts[7]

    # 클래스가 지정된 목록에 있는 경우에만 작업 수행
    if class_name in classes:
        # 새 경로를 만듭니다
        new_dir = f'./images/optical_window30_new/{class_name}/{course}/{event_class}/{count}'
        new_path = os.path.join(new_dir, filename)

        # 새 디렉토리가 없으면 만듭니다
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        # 파일을 새 경로로 이동시킵니다
        shutil.move(path, new_path)

print("지정된 클래스에 대한 파일 경로 재정렬 및 이동이 완료되었습니다.")

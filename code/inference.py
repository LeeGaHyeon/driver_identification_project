import re

# 로그 파일 경로
log_file_path = "/home/mmc/disk/driver_identification_old/logs/log_1021_cross_Aug2sampling_w30_C.txt" # mobaxterm 

# valid set1 loss와 관련된 데이터 저장 변수
min_valid_loss = float('inf')
min_vote_3 = None
min_vote_10 = None
min_vote_all = None
min_val_acc = None  # 추가된 변수

# 로그 파일 읽기
with open(log_file_path, 'r') as log_file:
    lines = log_file.readlines()
    for i in range(len(lines)):
        line = lines[i]
        # valid set1 loss와 val_acc를 추출하기 위한 정규식
        loss_val_acc_match = re.search(r'valid set1 loss: (\d+\.\d+).*val_acc: (\d+\.\d+)', line)
        
        if loss_val_acc_match:
            valid_loss = float(loss_val_acc_match.group(1))
            val_acc = float(loss_val_acc_match.group(2))
            
            # 최소 valid set1 loss 갱신 
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                min_val_acc = val_acc
                
                # 다음 라인에서 vote 값 추출
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    vote_match = re.search(r'vote_3: (\d+\.\d+), vote_10: (\d+\.\d+), vote_all: (\d+\.\d+)', next_line)
                    
                    if vote_match:
                        min_vote_3 = float(vote_match.group(1))
                        min_vote_10 = float(vote_match.group(2))
                        min_vote_all = float(vote_match.group(3))

# 결과 출력
print(f'Minimum valid set1 loss: {min_valid_loss}')
print(f'val_acc: {min_val_acc}')
print(f'vote_3: {min_vote_3}')
print(f'vote_10: {min_vote_10}')
print(f'vote_all: {min_vote_all}')
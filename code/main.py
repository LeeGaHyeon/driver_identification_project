import os
import time
import argparse
from tqdm import tqdm
# import timm
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.util import *
from utils.sam import *
# import apex
# from apex import amp


from dataset import *
from model import *
from run import *
from pytorchtools import EarlyStopping
import matplotlib.pyplot as plt
Precautions_msg = '(주의사항) ---- \n'
import torchsummary
import torch, torch.nn as nn, torch.nn.functional as F
import natsort

args = parse_args()

# def fine_tuning_main(base, val_name, data_path, n=10, mi=True):
#     # 데이터셋 읽어오기
#     '''데이터셋 정리
#     인당 4번 주행
#     A코스: bump: 한 코스에 36개, corner: 한 코스에 13개
#     B코스: bump: 한 코스에 20개, corner: 한 코스에 6개
#     C코스: bump: 한 코스에 15개, corner: 한 코스에 14개
#     optical flow image는 한 영상(10초)에 299개

#     경로
#     video: data/leegihun_crop/A/bump/1/*.avi
#     optical: data/leegihun_optical/A/bump/1/*.jpg

#     ex)
#     leegihun_A_b_4_0of36_0001.jpg ~ leegihun_A_b_4_0of36_0299.jpg
#     leegihun_A_b_4_0of36.avi
#     '''
#     # auto_drive\data\visual_window_image\jojeongdeok\B\bump\1\img
#     if base == 'count':
#         train_path = []
#         val_path = []
#         for i in range(len(data_path)):
#             if data_path[i].split('/')[-2] == val_name:
#                 val_path.append(data_path[i])
#             else:
#                 train_path.append(data_path[i])

#         train_path = natsort.natsorted(train_path)
#         val_path = natsort.natsorted(val_path)

#         train_path = [train_path[i * n:(i + 1) * n] for i in range((len(train_path) + n - 1) // n)]
#         val_path = [val_path[i * n:(i + 1) * n] for i in range((len(val_path) + n - 1) // n)]

#         print('------train_data--------')
#         print(len(train_path))

#         transforms_train, transforms_mi, transforms_val = get_transforms(args.image_size)
#         # 데이터셋 읽어오기
#         if mi:
#             dataset_train = Finetuning_CustomDataset(train_path, 'train', args.image_size, transform=transforms_mi)
#             dataset_valid_set = Finetuning_CustomDataset(val_path, 'valid', args.image_size, transform=transforms_mi)
#         else:
#             dataset_train = Finetuning_CustomDataset(train_path, 'train', args.image_size, transform=transforms_train)
#             dataset_valid_set = Finetuning_CustomDataset(val_path, 'valid', args.image_size, transform=transforms_train)

#         train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
#                                                    num_workers=args.num_workers, shuffle=True)
#         valid_loader_set = torch.utils.data.DataLoader(dataset_valid_set, batch_size=args.batch_size,
#                                                        num_workers=args.num_workers)

#         return train_loader, valid_loader_set, val_name

#     if base == 'course':
#         train_path = []
#         val_path = []
#         # auto_drive\data\visual_window_image\jojeongdeok\B\bump\1\img
#         for i in range(len(data_path)):
#             if data_path[i].split('/')[-4] == val_name:
#                 val_path.append(data_path[i])
#             else:
#                 train_path.append(data_path[i])

#         train_path = natsort.natsorted(train_path)
#         val_path = natsort.natsorted(val_path)

#         train_path = [train_path[i * n:(i + 1) * n] for i in range((len(train_path) + n - 1) // n)]
#         val_path = [val_path[i * n:(i + 1) * n] for i in range((len(val_path) + n - 1) // n)]

#         print('------train_data--------')
#         print(len(train_path))
#         transforms_train, transforms_mi, transforms_val = get_transforms(args.image_size)

#         if mi:
#             dataset_train = Finetuning_CustomDataset(train_path, 'train', args.image_size, transform=transforms_mi)
#             dataset_valid_set = Finetuning_CustomDataset(val_path, 'valid', args.image_size, transform=transforms_mi)
#         else:
#             dataset_train = Finetuning_CustomDataset(train_path, 'train', args.image_size, transform=transforms_train)
#             dataset_valid_set = Finetuning_CustomDataset(val_path, 'valid', args.image_size, transform=transforms_train)

#         train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
#                                                    num_workers=args.num_workers, shuffle=True)
#         valid_loader_set = torch.utils.data.DataLoader(dataset_valid_set, batch_size=args.batch_size,
#                                                        num_workers=args.num_workers)
#         return train_loader, valid_loader_set, val_name


def fine_tuning_main(base, val_name, data_path, n=30, mi=True): 
    if base == 'count':
        train_path = []
        val_path = []
        for i in range(len(data_path)):
            if data_path[i].split('/')[-2] == val_name:
                val_path.append(data_path[i])
            else:
                train_path.append(data_path[i])

        train_path = natsort.natsorted(train_path)
        val_path = natsort.natsorted(val_path)

        train_path = [train_path[i * n:(i + 1) * n] for i in range((len(train_path) + n - 1) // n)]
        val_path = [val_path[i * n:(i + 1) * n] for i in range((len(val_path) + n - 1) // n)]

        print('------train_data--------')
        print(len(train_path))

        transforms_train, transforms_mi, transforms_val = get_transforms(args.image_size) 
        if mi:
            dataset_train = Finetuning_CustomDataset(train_path, 'train', args.image_size, transform=transforms_mi)
            dataset_valid_set = Finetuning_CustomDataset(val_path, 'valid', args.image_size, transform=transforms_mi)
        else:
            dataset_train = Finetuning_CustomDataset(train_path, 'train', args.image_size, transform=transforms_train)
            dataset_valid_set = Finetuning_CustomDataset(val_path, 'valid', args.image_size, transform=transforms_train)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, shuffle=True)
        valid_loader_set = torch.utils.data.DataLoader(dataset_valid_set, batch_size=args.batch_size,
                                                       num_workers=args.num_workers)

        return train_loader, valid_loader_set, val_name



#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################

# def cross_main(base, val_name, visual, optical, n=10):
#     if base == 'count':
#         train_visual_path = []
#         val_visual_path = []
#
#         train_optical_path = []
#         val_optical_path = []
#         # auto_drive\data\visual_window_image\jojeongdeok\B\bump\1\img
#         for i in range(len(optical)):
#             if visual[i].split('\\')[-2] == val_name:
#                 val_visual_path.append(visual[i])
#             else:
#                 train_visual_path.append(visual[i])
#
#             if optical[i].split('\\')[-2] == val_name:
#                 val_optical_path.append(optical[i])
#             else:
#                 train_optical_path.append(optical[i])
#
#         train_visual_path = natsort.natsorted(train_visual_path)
#         val_visual_path = natsort.natsorted(val_visual_path)
#         train_optical_path = natsort.natsorted(train_optical_path)
#         val_optical_path = natsort.natsorted(val_optical_path)
#
#         train_visual_path = [train_visual_path[i * n:(i + 1) * n] for i in range((len(train_visual_path) + n - 1) // n)]
#         val_visual_path = [val_visual_path[i * n:(i + 1) * n] for i in range((len(val_visual_path) + n - 1) // n)]
#
#         train_optical_path = [train_optical_path[i * n:(i + 1) * n] for i in range((len(train_optical_path) + n - 1) // n)]
#         val_optical_path = [val_optical_path[i * n:(i + 1) * n] for i in range((len(val_optical_path) + n - 1) // n)]
#         print('------train_data--------')
#         print(len(train_visual_path))
#
#         transforms_train, transforms_val = get_transforms(args.image_size)
#         # 데이터셋 읽어오기
#         dataset_train = Cross_CustomDataset(train_visual_path, train_optical_path, 'train', args.image_size, transform=transforms_train)
#         dataset_valid_set = Cross_CustomDataset(val_visual_path, val_optical_path, 'valid', args.image_size, transform=transforms_val)
#
#         train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
#                                                    num_workers=args.num_workers, shuffle=True)
#         valid_loader_set = torch.utils.data.DataLoader(dataset_valid_set, batch_size=args.batch_size,
#                                                        num_workers=args.num_workers)
#
#         return train_loader, valid_loader_set, val_name
#
#     if base == 'course':
#         train_visual_path = []
#         val_visual_path = []
#
#         train_optical_path = []
#         val_optical_path = []
#         # auto_drive\data\visual_window_image\jojeongdeok\B\bump\1\img
#         for i in range(len(optical)):
#             if visual[i].split('\\')[-4] == val_name:
#                 val_visual_path.append(visual[i])
#             else:
#                 train_visual_path.append(visual[i])
#
#             if optical[i].split('\\')[-4] == val_name:
#                 val_optical_path.append(optical[i])
#             else:
#                 train_optical_path.append(optical[i])
#
#         train_visual_path = natsort.natsorted(train_visual_path)
#         val_visual_path = natsort.natsorted(val_visual_path)
#         train_optical_path = natsort.natsorted(train_optical_path)
#         val_optical_path = natsort.natsorted(val_optical_path)
#
#         train_visual_path = [train_visual_path[i * n:(i + 1) * n] for i in range((len(train_visual_path) + n - 1) // n)]
#         val_visual_path = [val_visual_path[i * n:(i + 1) * n] for i in range((len(val_visual_path) + n - 1) // n)]
#
#         train_optical_path = [train_optical_path[i * n:(i + 1) * n] for i in
#                               range((len(train_optical_path) + n - 1) // n)]
#         val_optical_path = [val_optical_path[i * n:(i + 1) * n] for i in range((len(val_optical_path) + n - 1) // n)]
#         print('------train_data--------')
#         print(len(train_visual_path))
#
#         transforms_train, transforms_val = get_transforms(args.image_size)
#         # 데이터셋 읽어오기
#         dataset_train = Cross_CustomDataset(train_visual_path, train_optical_path, 'train', args.image_size,
#                                             transform=transforms_train)
#         dataset_valid_set = Cross_CustomDataset(val_visual_path, val_optical_path, 'valid', args.image_size,
#                                                 transform=transforms_val)
#
#         train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
#                                                    num_workers=args.num_workers, shuffle=True)
#         valid_loader_set = torch.utils.data.DataLoader(dataset_valid_set, batch_size=args.batch_size,
#                                                        num_workers=args.num_workers)
#         # run(train_loader, valid_loader_set, course[j])
#         return train_loader, valid_loader_set, val_name
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################


def video_main(base, val_name, data_path):
    # a = torchvision.io.read_video(data_path, pts_unit='sec')[0][0::3] 
    if base == 'count':
        train_path = []
        val_path = []
        # auto_drive\data\visual_window_image\jojeongdeok\B\bump\1\img 
        for i in range(len(data_path)):
            if data_path[i].split('\\')[-2] == val_name:
                val_path.append(data_path[i])
            else:
                train_path.append(data_path[i])

        train_path = natsort.natsorted(train_path)
        val_path = natsort.natsorted(val_path)

        print('------train_data--------')
        print(len(train_path))
        print(len(val_path))

        transforms_train, transforms_mi, transforms_val = get_transforms(args.image_size)
        # 데이터셋 읽어오기
        dataset_train = Video_CustomDataset(train_path, 'train', args.image_size, transform=transforms_train)
        dataset_valid_set = Video_CustomDataset(val_path, 'valid', args.image_size, transform=transforms_val)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, shuffle=True)
        valid_loader_set = torch.utils.data.DataLoader(dataset_valid_set, batch_size=args.batch_size,
                                                       num_workers=args.num_workers)

        # run(train_loader, valid_loader_set, count[j])
        return train_loader, valid_loader_set, val_name
    if base == 'course':
        train_path = []
        val_path = [] 
        # auto_drive\data\visual_window_image\jojeongdeok\B\bump\1\img
        for i in range(len(data_path)):
            if data_path[i].split('\\')[-4] == val_name:
                val_path.append(data_path[i])
            else:
                train_path.append(data_path[i])

        train_path = natsort.natsorted(train_path)
        val_path = natsort.natsorted(val_path) 

        print('------train_data--------')
        print(len(train_path))
        print(len(val_path))
        transforms_train, transforms_mi, transforms_val = get_transforms(args.image_size)

        dataset_train = Video_CustomDataset(train_path, 'train', args.image_size, transform=transforms_train)
        dataset_valid_set = Video_CustomDataset(val_path, 'valid', args.image_size, transform=transforms_val)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, shuffle=True)
        valid_loader_set = torch.utils.data.DataLoader(dataset_valid_set, batch_size=args.batch_size,
                                                       num_workers=args.num_workers)
        return train_loader, valid_loader_set, val_name
        # run(train_loader, valid_loader_set, course[j])


#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################


# def cross_main(base, val_name, visual, optical, n=10):
#     if base == 'count':
#         train_visual_path = []
#         val_visual_path = []

#         train_optical_path = []
#         val_optical_path = []
    
#         # auto_drive\data\visual_window_image\jojeongdeok\B\bump\1\img
        
#         for i in range(len(optical)):   
#             print(visual[i])
#             if visual[i].split('/')[-2] == val_name: 
#                 val_visual_path.append(visual[i])
#             else:
#                 train_visual_path.append(visual[i])

#             if optical[i].split('/')[-2] == val_name:
#                 val_optical_path.append(optical[i])
#             else:
#                 train_optical_path.append(optical[i])
        
#         # for i in range(len(optical)):   
#         #     try:
#         #         # visual path 처리
#         #         if visual[i].split('/')[-2] == val_name: 
#         #             val_visual_path.append(visual[i])
#         #         else:
#         #             train_visual_path.append(visual[i])

#         #         # optical path 처리
#         #         if optical[i].split('/')[-2] == val_name:
#         #             val_optical_path.append(optical[i])
#         #         else:
#         #             train_optical_path.append(optical[i])

#         #         # 중간 출력
#         #         print(f"Iteration {i}:")
#         #         print(f"  train_visual_path count: {len(train_visual_path)}")
#         #         print(f"  val_visual_path count: {len(val_visual_path)}")
#         #         print(f"  train_optical_path count: {len(train_optical_path)}")
#         #         print(f"  val_optical_path count: {len(val_optical_path)}")

#         #     except Exception as e:
#         #         print(f"Error at iteration {i}: {e}")
#         #         break  # 에러 발생 시 루프를 종료하거나 로그를 추가

#         train_visual_path = natsort.natsorted(train_visual_path) 
#         val_visual_path = natsort.natsorted(val_visual_path)
#         train_optical_path = natsort.natsorted(train_optical_path)
#         val_optical_path = natsort.natsorted(val_optical_path)

#         train_visual_path = [train_visual_path[i * n:(i + 1) * n] for i in range((len(train_visual_path) + n - 1) // n)]
#         val_visual_path = [val_visual_path[i * n:(i + 1) * n] for i in range((len(val_visual_path) + n - 1) // n)]

#         train_optical_path = [train_optical_path[i * n:(i + 1) * n] for i in range((len(train_optical_path) + n - 1) // n)]
#         val_optical_path = [val_optical_path[i * n:(i + 1) * n] for i in range((len(val_optical_path) + n - 1) // n)]
#         print('------train_data--------')
#         print(len(train_visual_path))

#         transforms_train, transforms_mi, transforms_val = get_transforms(args.image_size)
#         # 데이터셋 읽어오기
#         dataset_train = Cross_CustomDataset(train_visual_path, train_optical_path, 'train', args.image_size, transform=transforms_train, mi_transform=transforms_mi)
#         dataset_valid_set = Cross_CustomDataset(val_visual_path, val_optical_path, 'valid', args.image_size, transform=transforms_val, mi_transform=transforms_val)

#         train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
#                                                    num_workers=args.num_workers, shuffle=True)
#         valid_loader_set = torch.utils.data.DataLoader(dataset_valid_set, batch_size=args.batch_size,
#                                                        num_workers=args.num_workers)

#         return train_loader, valid_loader_set, val_name

#     if base == 'course':
#         train_visual_path = []
#         val_visual_path = []

#         train_optical_path = []
#         val_optical_path = []
#         # auto_drive\data\visual_window_image\jojeongdeok\B\bump\1\img 
#         for i in range(len(optical)):
#             if visual[i].split('/')[-4] == val_name:
#                 val_visual_path.append(visual[i])
#             else:
#                 train_visual_path.append(visual[i])

#             if optical[i].split('/')[-4] == val_name:
#                 val_optical_path.append(optical[i])
#             else:
#                 train_optical_path.append(optical[i])

#         train_visual_path = natsort.natsorted(train_visual_path)
#         val_visual_path = natsort.natsorted(val_visual_path)
#         train_optical_path = natsort.natsorted(train_optical_path)
#         val_optical_path = natsort.natsorted(val_optical_path)

#         train_visual_path = [train_visual_path[i * n:(i + 1) * n] for i in range((len(train_visual_path) + n - 1) // n)]
#         val_visual_path = [val_visual_path[i * n:(i + 1) * n] for i in range((len(val_visual_path) + n - 1) // n)]

#         train_optical_path = [train_optical_path[i * n:(i + 1) * n] for i in
#                               range((len(train_optical_path) + n - 1) // n)]
#         val_optical_path = [val_optical_path[i * n:(i + 1) * n] for i in range((len(val_optical_path) + n - 1) // n)]
#         print('------train_data--------')
#         print(len(train_visual_path))

#         transforms_train, transforms_mi, transforms_val = get_transforms(args.image_size)
#         # 데이터셋 읽어오기
#         dataset_train = Cross_CustomDataset(train_visual_path, train_optical_path, 'train', args.image_size,
#                                             transform=transforms_train, mi_transform=transforms_mi)
#         dataset_valid_set = Cross_CustomDataset(val_visual_path, val_optical_path, 'valid', args.image_size,
#                                                 transform=transforms_val, mi_transform=transforms_val)

#         # Sensor_CustomDataset
#         train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
#                                                    num_workers=args.num_workers, shuffle=True)
#         valid_loader_set = torch.utils.data.DataLoader(dataset_valid_set, batch_size=args.batch_size,
#                                                        num_workers=args.num_workers)
#         # run(train_loader, valid_loader_set, course[j])
#         return train_loader, valid_loader_set, val_name


def cross_main(base, val_name, visual, optical, n=10):
    if base == 'count':
        train_visual_path = []
        val_visual_path = []

        train_optical_path = []
        val_optical_path = []
    
        # auto_drive\data\visual_window_image\jojeongdeok\B\bump\1\img
        
        for i in range(len(optical)):    
            if visual[i].split('/')[4] == val_name: 
                val_visual_path.append(visual[i])
            else:
                train_visual_path.append(visual[i])

            if optical[i].split('/')[4] == val_name:
                val_optical_path.append(optical[i])
            else:
                train_optical_path.append(optical[i])

        train_visual_path = natsort.natsorted(train_visual_path) 
        val_visual_path = natsort.natsorted(val_visual_path)
        train_optical_path = natsort.natsorted(train_optical_path)
        val_optical_path = natsort.natsorted(val_optical_path)

        train_visual_path = [train_visual_path[i * n:(i + 1) * n] for i in range((len(train_visual_path) + n - 1) // n)]
        val_visual_path = [val_visual_path[i * n:(i + 1) * n] for i in range((len(val_visual_path) + n - 1) // n)]

        train_optical_path = [train_optical_path[i * n:(i + 1) * n] for i in range((len(train_optical_path) + n - 1) // n)]
        val_optical_path = [val_optical_path[i * n:(i + 1) * n] for i in range((len(val_optical_path) + n - 1) // n)] 
        print('------train_data--------')
        print(len(train_visual_path)) 
        print(len(val_optical_path))
        transforms_train, transforms_mi, transforms_val = get_transforms(args.image_size)
        # 데이터셋 읽어오기
        dataset_train = Cross_CustomDataset(train_visual_path, train_optical_path, 'train', args.image_size, transform=transforms_train, mi_transform=transforms_mi)
        dataset_valid_set = Cross_CustomDataset(val_visual_path, val_optical_path, 'valid', args.image_size, transform=transforms_val, mi_transform=transforms_val)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, shuffle=True)
        valid_loader_set = torch.utils.data.DataLoader(dataset_valid_set, batch_size=args.batch_size,
                                                       num_workers=args.num_workers) 

        return train_loader, valid_loader_set, val_name


def cross_sensor_main(base, val_name, visual, optical, n=10):
    if base == 'count':
        train_visual_path = []
        val_visual_path = []

        train_optical_path = []
        val_optical_path = []
        # auto_drive\data\visual_window_image\jojeongdeok\B\bump\1\img
        for i in range(len(optical)):
            if visual[i].split('\\')[-2] == val_name:
                val_visual_path.append(visual[i])
            else:
                train_visual_path.append(visual[i])

            if optical[i].split('\\')[-2] == val_name:
                val_optical_path.append(optical[i])
            else:
                train_optical_path.append(optical[i])

        train_visual_path = natsort.natsorted(train_visual_path)
        val_visual_path = natsort.natsorted(val_visual_path)
        train_optical_path = natsort.natsorted(train_optical_path)
        val_optical_path = natsort.natsorted(val_optical_path)

        train_visual_path = [train_visual_path[i * n:(i + 1) * n] for i in range((len(train_visual_path) + n - 1) // n)]
        val_visual_path = [val_visual_path[i * n:(i + 1) * n] for i in range((len(val_visual_path) + n - 1) // n)]

        train_optical_path = [train_optical_path[i * n:(i + 1) * n] for i in
                              range((len(train_optical_path) + n - 1) // n)]
        val_optical_path = [val_optical_path[i * n:(i + 1) * n] for i in range((len(val_optical_path) + n - 1) // n)]
        print('------train_data--------')
        print(len(train_visual_path))

        transforms_train, transforms_mi, transforms_val = get_transforms(args.image_size)
        # 데이터셋 읽어오기
        dataset_train = Sensor_CustomDataset(train_visual_path, train_optical_path, 'train', args.image_size,
                                            transform=transforms_train, mi_transform=transforms_mi)
        dataset_valid_set = Sensor_CustomDataset(val_visual_path, val_optical_path, 'valid', args.image_size,
                                                transform=transforms_val, mi_transform=transforms_val)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, shuffle=True)
        valid_loader_set = torch.utils.data.DataLoader(dataset_valid_set, batch_size=args.batch_size,
                                                       num_workers=args.num_workers)

        return train_loader, valid_loader_set, val_name

    if base == 'course':
        train_visual_path = []
        val_visual_path = []

        train_optical_path = []
        val_optical_path = []
        # auto_drive\data\visual_window_image\jojeongdeok\B\bump\1\img
        for i in range(len(optical)):
            if visual[i].split('\\')[-4] == val_name:
                val_visual_path.append(visual[i])
            else:
                train_visual_path.append(visual[i])

            if optical[i].split('\\')[-4] == val_name:
                val_optical_path.append(optical[i])
            else:
                train_optical_path.append(optical[i])

        train_visual_path = natsort.natsorted(train_visual_path)
        val_visual_path = natsort.natsorted(val_visual_path)
        train_optical_path = natsort.natsorted(train_optical_path)
        val_optical_path = natsort.natsorted(val_optical_path)

        train_visual_path = [train_visual_path[i * n:(i + 1) * n] for i in range((len(train_visual_path) + n - 1) // n)]
        val_visual_path = [val_visual_path[i * n:(i + 1) * n] for i in range((len(val_visual_path) + n - 1) // n)]

        train_optical_path = [train_optical_path[i * n:(i + 1) * n] for i in
                              range((len(train_optical_path) + n - 1) // n)]
        val_optical_path = [val_optical_path[i * n:(i + 1) * n] for i in range((len(val_optical_path) + n - 1) // n)]
        print('------train_data--------')
        print(len(train_visual_path))

        transforms_train, transforms_mi, transforms_val = get_transforms(args.image_size)
        # 데이터셋 읽어오기
        dataset_train = Sensor_CustomDataset(train_visual_path, train_optical_path, 'train', args.image_size,
                                            transform=transforms_train, mi_transform=transforms_mi)
        dataset_valid_set = Sensor_CustomDataset(val_visual_path, val_optical_path, 'valid', args.image_size,
                                                transform=transforms_val, mi_transform=transforms_val)

        # Sensor_CustomDataset
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, shuffle=True)
        valid_loader_set = torch.utils.data.DataLoader(dataset_valid_set, batch_size=args.batch_size,
                                                       num_workers=args.num_workers)
        # run(train_loader, valid_loader_set, course[j])
        return train_loader, valid_loader_set, val_name


def cross_auto_main(base, val_name, visual, optical, n=10):
    if base == 'count':
        train_visual_path = []
        val_visual_path = []

        train_optical_path = []
        val_optical_path = []
        #auto/window/name/1/img
        for i in range(len(optical)):
            if visual[i].split('\\')[-2] == val_name:
                val_visual_path.append(visual[i])
            else:
                train_visual_path.append(visual[i])

            if optical[i].split('\\')[-2] == val_name:
                val_optical_path.append(optical[i])
            else:
                train_optical_path.append(optical[i])

        train_visual_path = natsort.natsorted(train_visual_path)
        val_visual_path = natsort.natsorted(val_visual_path)
        train_optical_path = natsort.natsorted(train_optical_path)
        val_optical_path = natsort.natsorted(val_optical_path)

        train_visual_path = [train_visual_path[i * n:(i + 1) * n] for i in range((len(train_visual_path) + n - 1) // n)]
        val_visual_path = [val_visual_path[i * n:(i + 1) * n] for i in range((len(val_visual_path) + n - 1) // n)]

        train_optical_path = [train_optical_path[i * n:(i + 1) * n] for i in
                              range((len(train_optical_path) + n - 1) // n)]
        val_optical_path = [val_optical_path[i * n:(i + 1) * n] for i in range((len(val_optical_path) + n - 1) // n)]
        print('------train_data--------')
        print(len(train_visual_path))

        transforms_train, transforms_mi, transforms_val = get_transforms(args.image_size)
        # 데이터셋 읽어오기
        dataset_train = Auto_CustomDataset(train_visual_path, train_optical_path, 'train', args.image_size,
                                            transform=transforms_train, mi_transform=transforms_mi)
        dataset_valid_set = Auto_CustomDataset(val_visual_path, val_optical_path, 'valid', args.image_size,
                                                transform=transforms_val, mi_transform=transforms_val)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, shuffle=True)
        valid_loader_set = torch.utils.data.DataLoader(dataset_valid_set, batch_size=args.batch_size,
                                                       num_workers=args.num_workers)

        return train_loader, valid_loader_set, val_name

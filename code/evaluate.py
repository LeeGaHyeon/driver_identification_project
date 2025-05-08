import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, average_precision_score  # 수정된 부분
import argparse
import random
from sklearn.preprocessing import label_binarize

import os
import torch
import pandas as pd
import argparse 
from tqdm import tqdm
from dataset import *  
from model import * 
from utils.util import *
from main import *
import os
import torch
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
import argparse
import random
from sklearn.preprocessing import label_binarize

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--image-size', type=int, default='256')
    parser.add_argument('--enet-type', type=str, required=True, default='tf_efficientnet_b0_ns')
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--n-meta-dim', type=str, default='512,256')
    parser.add_argument('--out-dim', type=int, default=15)
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='5')
    parser.add_argument('--k-fold', type=int, default=5)
    parser.add_argument('--log-dir', type=str, default='./prob')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()
    return args

from sklearn.metrics import precision_recall_curve
import numpy as np

def calculate_metrics(all_targets, all_probs_np, num_classes):
    preds = np.argmax(all_probs_np, axis=1)
    accuracy = accuracy_score(all_targets, preds)
    f1 = f1_score(all_targets, preds, average='macro')

    # Top-3 Accuracy Calculation
    top3_preds = np.argsort(all_probs_np, axis=1)[:, -3:]
    top3_accuracy = np.mean([all_targets[i] in top3_preds[i] for i in range(len(all_targets))])

    # Mean Average Precision (mAP) Calculation
    all_targets_bin = label_binarize(all_targets, classes=np.arange(num_classes))
    average_precisions = []
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(all_targets_bin[:, i], all_probs_np[:, i])
        ap = np.trapz(recall, precision)
        average_precisions.append(ap)
    map_score = np.mean(average_precisions)

    return accuracy, f1, top3_accuracy, map_score


def calculate_voting_metrics(logit_list, target_list, num_classes):
    voting_results = {'voting_1': {}, 'voting_3': {}, 'voting_10': {}, 'vote_all': {}}

    for voting_type in ['voting_1', 'voting_3', 'voting_10', 'vote_all']:
        all_probs_voting = []
        all_targets_voting = []

        for i in range(num_classes):
            people_index = list(filter(lambda x: target_list[x] == i, range(len(target_list))))
            if len(people_index) == 0:
                continue

            if voting_type == 'voting_1':
                for idx in people_index:
                    all_probs_voting.append(logit_list[idx])
                    all_targets_voting.append(i)

            elif voting_type == 'voting_3':
                for x in range(0, len(people_index), 3):
                    softmax_3 = sum([logit_list[j] for j in people_index[x:x + 3]]) / min(3, len(people_index) - x)
                    all_probs_voting.append(softmax_3)
                    all_targets_voting.append(i)

            elif voting_type == 'voting_10':
                for x in range(0, len(people_index), 10):
                    softmax_10 = sum([logit_list[j] for j in people_index[x:x + 10]]) / min(10, len(people_index) - x)
                    all_probs_voting.append(softmax_10)
                    all_targets_voting.append(i)

            elif voting_type == 'vote_all':
                softmax_all = sum([logit_list[x] for x in people_index]) / len(people_index)
                all_probs_voting.append(softmax_all)
                all_targets_voting.append(i)

        all_probs_voting_np = np.array(all_probs_voting)
        all_targets_voting_np = np.array(all_targets_voting)

        accuracy, f1, top3_accuracy, map_score = calculate_metrics(all_targets_voting_np, all_probs_voting_np, num_classes)

        voting_results[voting_type]['accuracy'] = accuracy
        voting_results[voting_type]['f1'] = f1
        voting_results[voting_type]['top3_accuracy'] = top3_accuracy
        voting_results[voting_type]['map_score'] = map_score

    return voting_results

def run_model_and_collect_logits(model, dataloader, device, num_classes=15):
    model.eval()
    all_targets = []
    logit_list = []
    target_list = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            org_images, optical_images, labels = batch[:3]
            org_images = org_images.to(device)
            optical_images = optical_images.to(device)
            labels = labels.to(device)

            logits = model(org_images, optical_images, device)
            probs = torch.nn.functional.softmax(logits, dim=1)

            logit_list.extend(probs.cpu().numpy())
            target_list.extend(labels.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    return logit_list, target_list, all_targets

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device=="cuda": 
        print('cuda')
        torch.cuda.empty_cache()
        print('empty')
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    set_seed(4922)

    # 각 코스에 대한 정확도 계산
    course = ['A', 'B', 'C']
    count = ['1', '2', '3', '4']
    visual = natsort.natsorted(glob.glob('./images/visual_window30/*****/****/***/**/*.jpg'))
    optical = natsort.natsorted(glob.glob('./images/optical_window30/*****/****/***/**/*.jpg'))
    n = 30

    # 각 VOTING 방식별 결과 저장
    voting_types = ['voting_1', 'voting_3', 'voting_10', 'vote_all']
    voting_results_avg_course = {v: {'accuracy': [], 'f1': [], 'top3_accuracy': [], 'map_score': []} for v in voting_types}
    voting_results_avg_round = {v: {'accuracy': [], 'f1': [], 'top3_accuracy': [], 'map_score': []} for v in voting_types}

    # 코스별 성능 계산
    for j in range(len(course)):
        train_loader, valid_loader_set, valid_name = cross_main('course', course[j], visual, optical, n)

        # 모델 로드
        model_file = os.path.join(args.model_dir, f'0902_AugAug2Cross_w30_best_fold_{valid_name}.pth.pth')
        if os.path.isfile(model_file):
            model = Cross_model(num_frames=n, out_dim=args.out_dim, valid_name=valid_name)
            model.load_state_dict(torch.load(model_file))
            model = model.to(device)
        else:
            print(f"Model file {model_file} not found.")
            continue

        logit_list, target_list, all_targets = run_model_and_collect_logits(model, valid_loader_set, device)

        voting_results = calculate_voting_metrics(logit_list, target_list, args.out_dim)

        print(f"Results for course {course[j]}:")
        for voting_type, results in voting_results.items():
            print(f"  {voting_type}:")
            print(f"    Accuracy: {results['accuracy']*100:.2f}, F1: {results['f1']*100:.2f}, Top-3 Accuracy: {results['top3_accuracy']*100:.2f}, mAP: {results['map_score']*100:.2f}")

            voting_results_avg_course[voting_type]['accuracy'].append(results['accuracy']*100)
            voting_results_avg_course[voting_type]['f1'].append(results['f1']*100)
            voting_results_avg_course[voting_type]['top3_accuracy'].append(results['top3_accuracy']*100)
            voting_results_avg_course[voting_type]['map_score'].append(results['map_score']*100)

    print("\nAverage results by voting type for courses:")
    for voting_type in voting_types:
        print(f"  {voting_type}:")
        print(f"    Accuracy: {np.mean(voting_results_avg_course[voting_type]['accuracy']):.2f}")
        print(f"    F1: {np.mean(voting_results_avg_course[voting_type]['f1']):.2f}")
        print(f"    Top-3 Accuracy: {np.mean(voting_results_avg_course[voting_type]['top3_accuracy']):.2f}")
        print(f"    mAP: {np.mean(voting_results_avg_course[voting_type]['map_score']):.2f}")

    # for j in range(len(count)):
    #     train_loader, valid_loader_set, valid_name = cross_main('count', count[j], visual, optical, n)

    #     model_file = os.path.join(args.model_dir, f'0902_AugAug2Cross_w30_best_fold_{valid_name}.pth.pth')
    #     if os.path.isfile(model_file):
    #         model = Cross_model(num_frames=n, out_dim=args.out_dim, valid_name=valid_name)
    #         model.load_state_dict(torch.load(model_file))
    #         model = model.to(device)
    #     else:
    #         print(f"Model file {model_file} not found.")
    #         continue

    #     logit_list, target_list, all_targets = run_model_and_collect_logits(model, valid_loader_set, device)

    #     voting_results = calculate_voting_metrics(logit_list, target_list, args.out_dim)

    #     print(f"Results for round {count[j]}:")
    #     for voting_type, results in voting_results.items():
    #         print(f"  {voting_type}:")
    #         print(f"    Accuracy: {results['accuracy']*100:.2f}, F1: {results['f1']*100:.2f}, Top-3 Accuracy: {results['top3_accuracy']*100:.2f}, mAP: {results['map_score']*100:.2f}")

    #         voting_results_avg_round[voting_type]['accuracy'].append(results['accuracy']*100)
    #         voting_results_avg_round[voting_type]['f1'].append(results['f1']*100)
    #         voting_results_avg_round[voting_type]['top3_accuracy'].append(results['top3_accuracy']*100)
    #         voting_results_avg_round[voting_type]['map_score'].append(results['map_score']*100)

    # print("\nAverage results by voting type for rounds:")
    # for voting_type in voting_types:
    #     print(f"  {voting_type}:")
    #     print(f"    Accuracy: {np.mean(voting_results_avg_round[voting_type]['accuracy']):.2f}")
    #     print(f"    F1: {np.mean(voting_results_avg_round[voting_type]['f1']):.2f}")
    #     print(f"    Top-3 Accuracy: {np.mean(voting_results_avg_round[voting_type]['top3_accuracy']):.2f}")
    #     print(f"    mAP: {np.mean(voting_results_avg_round[voting_type]['map_score']):.2f}")

if __name__ == '__main__': 
    main()

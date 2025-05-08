import pandas as pd
from sklearn.metrics import f1_score, average_precision_score

def ensemble_metrics(rounds=4):
    total_accuracy = 0
    total_top3_accuracy = 0
    total_f1_score = 0
    total_map_score = 0

    for round_num in range(1, rounds + 1):
        vi_round_avg_C = pd.read_csv(f'./prob/0902_AugAug2Cross_w30_logits_{round_num}.csv')
        sensor_round_C_avg_prob = pd.read_csv(f'./prob/sensor_round_{round_num}_avg_prob.csv')
        can_round_C_avg_prob = pd.read_csv(f'./prob/can_round_{round_num}_avg_prob.csv')

        val_col = f'val_{round_num}'
        merged_df = vi_round_avg_C.merge(sensor_round_C_avg_prob, on=val_col, suffixes=('_vi', '_sensor'))
        merged_df = merged_df.merge(can_round_C_avg_prob, on=val_col, suffixes=('', '_Can'))

        label_columns = [str(i) for i in range(15)]
        for label in label_columns:
            merged_df[label + '_avg'] = (merged_df[label + '_vi'] + merged_df[label + '_sensor'] + merged_df[label]) / 3

        merged_df['predicted_label'] = merged_df[[label + '_avg' for label in label_columns]].idxmax(axis=1)
        merged_df['predicted_label'] = merged_df['predicted_label'].str.replace('_avg', '').astype(int)

        merged_df['correct'] = merged_df[val_col] == merged_df['predicted_label']
        accuracy = merged_df['correct'].mean()
        print(f"Round {round_num} - Accuracy: {accuracy * 100:.2f}%")

        merged_df['top_3_labels'] = merged_df[[label + '_avg' for label in label_columns]].apply(
            lambda row: row.nlargest(3).index.str.replace('_avg', '').astype(int).tolist(), axis=1)
        merged_df['top_3_correct'] = merged_df.apply(lambda row: row[val_col] in row['top_3_labels'], axis=1)
        top3_accuracy = merged_df['top_3_correct'].mean()
        print(f"Round {round_num} - Top-3 Accuracy: {top3_accuracy * 100:.2f}%")

        f1 = f1_score(merged_df[val_col], merged_df['predicted_label'], average='macro')
        print(f"Round {round_num} - F1-Score: {f1 * 100:.2f}%")

        # Calculate Mean Average Precision (mAP) using average_precision_score
        y_true = pd.get_dummies(merged_df[val_col], columns=label_columns)
        y_pred = merged_df[[label + '_avg' for label in label_columns]]
        map_score = average_precision_score(y_true, y_pred, average='macro')
        print(f"Round {round_num} - Mean Average Precision (mAP): {map_score * 100:.2f}%")

        total_accuracy += accuracy
        total_top3_accuracy += top3_accuracy
        total_f1_score += f1
        total_map_score += map_score

    avg_accuracy = total_accuracy / rounds
    avg_top3_accuracy = total_top3_accuracy / rounds
    avg_f1_score = total_f1_score / rounds
    avg_map_score = total_map_score / rounds

    print(f"\nAverage across {rounds} rounds:")
    print(f"Accuracy: {avg_accuracy * 100:.2f}%")
    print(f"Top-3 Accuracy: {avg_top3_accuracy * 100:.2f}%")
    print(f"F1-Score: {avg_f1_score * 100:.2f}%")
    print(f"Mean Average Precision (mAP): {avg_map_score * 100:.2f}%")

if __name__ == "__main__":
    ensemble_metrics()


# import pandas as pd
# from sklearn.metrics import f1_score, average_precision_score

# def ensemble_metrics(courses=['A', 'B', 'C']):
#     # Initialize variables to accumulate metrics for averaging later
#     total_accuracy = 0
#     total_top3_accuracy = 0
#     total_f1_score = 0
#     total_map_score = 0
#     total_courses = 0

#     # Loop over each course (A, B, C)
#     for course in courses:
#         # Load the three CSV files for the current course
#         vi_round_avg_C = pd.read_csv(f'./prob/0902_AugAug2Cross_w30_logits_{course}.csv')
#         sensor_round_C_avg_prob = pd.read_csv(f'./prob/sensor_course_{course}_avg_prob.csv')
#         can_round_C_avg_prob = pd.read_csv(f'./prob/can_course_{course}_avg_prob.csv')

#         # Merge the three dataframes on 'val_{course}' (e.g., val_A, val_B, val_C)
#         val_col = f'val_{course}'
#         merged_df = vi_round_avg_C.merge(sensor_round_C_avg_prob, on=val_col, suffixes=('_vi', '_sensor'))
#         merged_df = merged_df.merge(can_round_C_avg_prob, on=val_col, suffixes=('', '_can'))

#         # Average the probabilities for each label using the provided weights
#         vi_weight = 0.2
#         sensor_weight = 0.55
#         can_weight = 0.25

#         label_columns = [str(i) for i in range(15)]
#         for label in label_columns:
#             merged_df[label + '_weighted_avg'] = (merged_df[label + '_vi'] * vi_weight + 
#                                                   merged_df[label + '_sensor'] * sensor_weight + 
#                                                   merged_df[label] * can_weight)

#         # Determine the predicted label (label with the highest weighted average probability)
#         merged_df['predicted_label'] = merged_df[[label + '_weighted_avg' for label in label_columns]].idxmax(axis=1)
#         merged_df['predicted_label'] = merged_df['predicted_label'].str.replace('_weighted_avg', '').astype(int)

#         # Calculate the Accuracy
#         merged_df['correct'] = merged_df[val_col] == merged_df['predicted_label']
#         accuracy = merged_df['correct'].mean()
#         print(f"Course {course} - Accuracy: {accuracy * 100:.2f}%")

#         # Calculate the Top-3 accuracy
#         merged_df['top_3_labels'] = merged_df[[label + '_weighted_avg' for label in label_columns]].apply(
#             lambda row: row.nlargest(3).index.str.replace('_weighted_avg', '').astype(int).tolist(), axis=1)
#         merged_df['top_3_Correct'] = merged_df.apply(lambda row: row[val_col] in row['top_3_labels'], axis=1)
#         top3_accuracy = merged_df['top_3_Correct'].mean()
#         print(f"Course {course} - Top-3 Accuracy: {top3_accuracy * 100:.2f}%")

#         # Calculate the F1-Score (converted to percentage)
#         f1 = f1_score(merged_df[val_col], merged_df['predicted_label'], average='macro')
#         print(f"Course {course} - F1-Score: {f1 * 100:.2f}%")

#         # Calculate Mean Average Precision (mAP) using average_precision_score
#         y_true = pd.get_dummies(merged_df[val_col], columns=label_columns)
#         y_pred = merged_df[[label + '_weighted_avg' for label in label_columns]]
#         map_score = average_precision_score(y_true, y_pred, average='macro')
#         print(f"Course {course} - Mean Average Precision (mAP): {map_score * 100:.2f}%")

#         # Accumulate metrics for averaging
#         total_accuracy += accuracy
#         total_top3_accuracy += top3_accuracy
#         total_f1_score += f1
#         total_map_score += map_score
#         total_courses += 1

#     # Calculate average metrics across all courses
#     avg_accuracy = total_accuracy / total_courses
#     avg_top3_accuracy = total_top3_accuracy / total_courses
#     avg_f1_score = total_f1_score / total_courses
#     avg_map_score = total_map_score / total_courses

#     # Print the average metrics
#     print(f"\nAverage across all courses:")
#     print(f"Accuracy: {avg_accuracy * 100:.2f}%")
#     print(f"Top-3 Accuracy: {avg_top3_accuracy * 100:.2f}%")
#     print(f"F1-Score: {avg_f1_score * 100:.2f}%")
#     print(f"Mean Average Precision (mAP): {avg_map_score * 100:.2f}%")

# if __name__ == "__main__":
#     ensemble_metrics()

# import pandas as pd
# from sklearn.metrics import f1_score

# def ensemble_metrics(courses=['A', 'B', 'C']):
#     # Initialize variables to accumulate metrics for averaging later
#     total_accuracy = 0
#     total_top3_accuracy = 0
#     total_f1_score = 0
#     total_map_score = 0

#     # Loop over each course (A, B, C)
#     for course in courses:
#         # Load the three CSV files for the current course
#         vi_round_avg_C = pd.read_csv(f'./prob/0902_AugAug2Cross_w30_logits_{course}.csv')
#         sensor_round_C_avg_prob = pd.read_csv(f'./prob/sensor_course_{course}_avg_prob.csv')
#         can_round_C_avg_prob = pd.read_csv(f'./prob/can_course_{course}_avg_prob.csv')

#         # Merge the three dataframes on 'val_{course}' (e.g., val_A, val_B, val_C)
#         val_col = f'val_{course}'
#         merged_df = vi_round_avg_C.merge(sensor_round_C_avg_prob, on=val_col, suffixes=('_vi', '_sensor'))
#         merged_df = merged_df.merge(can_round_C_avg_prob, on=val_col, suffixes=('', '_Can'))

#         # Average the probabilities for each label
#         label_Columns = [str(i) for i in range(15)]
#         for label in label_Columns:
#             merged_df[label + '_avg'] = (merged_df[label + '_vi'] + merged_df[label + '_sensor'] + merged_df[label]) / 3

#         # Determine the predicted label (label with the highest average probability)
#         merged_df['predicted_label'] = merged_df[[label + '_avg' for label in label_Columns]].idxmax(axis=1)
#         merged_df['predicted_label'] = merged_df['predicted_label'].str.replace('_avg', '').astype(int)

#         # Calculate the Accuracy
#         merged_df['correct'] = merged_df[val_col] == merged_df['predicted_label']
#         accuracy = merged_df['correct'].mean()
#         print(f"Course {course} - Accuracy: {accuracy * 100:.2f}%")

#         # Calculate the Top-3 accuracy
#         merged_df['top_4_labels'] = merged_df[[label + '_avg' for label in label_Columns]].apply(
#             lambda row: row.nlargest(3).index.str.replace('_avg', '').astype(int).tolist(), axis=1)
#         merged_df['top_4_Correct'] = merged_df.apply(lambda row: row[val_col] in row['top_4_labels'], axis=1)
#         top3_accuracy = merged_df['top_4_Correct'].mean()
#         print(f"Course {course} - Top-3 Accuracy: {top3_accuracy * 100:.2f}%")

#         # Calculate the F1-Score (converted to percentage)
#         f1 = f1_score(merged_df[val_col], merged_df['predicted_label'], average='macro')
#         print(f"Course {course} - F1-Score: {f1 * 100:.2f}%")

#         # Calculate Mean Average Precision (mAP, converted to percentage)
#         def average_precision_Ct_k(row, k):
#             top_k_labels = row['top_4_labels'][:k]
#             try:
#                 rank = top_k_labels.index(row[val_col]) + 1  # Find the rank (1-based index)
#                 return 1.0 / rank  # Precision at the rank
#             except ValueError:
#                 return 0.0  # If the true label is not in the top_k, return 0.0

#         merged_df['ap_at_4'] = merged_df.apply(lambda row: average_precision_Ct_k(row, 3), axis=1)
#         map_score = merged_df['ap_at_4'].mean()
#         print(f"Course {course} - Mean Average Precision (mAP): {map_score * 100:.2f}%")

#         # Accumulate metrics for averaging
#         total_accuracy += accuracy
#         total_top3_accuracy += top3_accuracy
#         total_f1_score += f1
#         total_map_score += map_score

#     # Calculate average metrics across all courses
#     avg_accuracy = total_accuracy / len(courses)
#     avg_top3_accuracy = total_top3_accuracy / len(courses)
#     avg_f1_score = total_f1_score / len(courses)
#     avg_map_score = total_map_score / len(courses)

#     # Print the average metrics
#     print(f"\nAverage across all courses:")
#     print(f"Accuracy: {avg_accuracy * 100:.2f}%")
#     print(f"Top-3 Accuracy: {avg_top3_accuracy * 100:.2f}%")
#     print(f"F1-Score: {avg_f1_score * 100:.2f}%")
#     print(f"Mean Average Precision (mAP): {avg_map_score * 100:.2f}%")

# if __name__ == "__main__":
#     ensemble_metrics() 

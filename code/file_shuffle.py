# # -*- coding: utf-8 -*-

# import os
# import random

# def shuffle_pairs(base_dir):
#     # ���� optical_window30���� ����, �� ����� visual_window30�� �����մϴ�.
#     optical_dir = os.path.join(base_dir, 'optical_window30')
#     visual_dir = os.path.join(base_dir, 'visual_window30')
    
#     for person_id in os.listdir(optical_dir):
#         optical_person_path = os.path.join(optical_dir, person_id)
#         visual_person_path = os.path.join(visual_dir, person_id)
        
#         if os.path.isdir(optical_person_path) and os.path.isdir(visual_person_path):
#             for section in ['A', 'B', 'C']:
#                 optical_section_path = os.path.join(optical_person_path, section)
#                 visual_section_path = os.path.join(visual_person_path, section)
                
#                 if os.path.isdir(optical_section_path) and os.path.isdir(visual_section_path):
#                     for scenario in ['bump', 'corner']:
#                         optical_scenario_path = os.path.join(optical_section_path, scenario)
#                         visual_scenario_path = os.path.join(visual_section_path, scenario)
                        
#                         if os.path.isdir(optical_scenario_path) and os.path.isdir(visual_scenario_path):
#                             # Get all round folders (1, 2, 3, 4)
#                             round_folders = [folder for folder in os.listdir(optical_scenario_path) if os.path.isdir(os.path.join(optical_scenario_path, folder))]
                            
#                             # Shuffle the round folders
#                             shuffled_folders = round_folders[:]
#                             random.shuffle(shuffled_folders)
                            
#                             # Rename the folders in both optical and visual paths based on the shuffled list
#                             for original_name, new_name in zip(round_folders, shuffled_folders):
#                                 optical_original_path = os.path.join(optical_scenario_path, original_name)
#                                 optical_new_path = os.path.join(optical_scenario_path, f"{new_name}_temp")
#                                 visual_original_path = os.path.join(visual_scenario_path, original_name)
#                                 visual_new_path = os.path.join(visual_scenario_path, f"{new_name}_temp")
                                
#                                 os.rename(optical_original_path, optical_new_path)
#                                 os.rename(visual_original_path, visual_new_path)
                            
#                             # Remove the '_temp' suffix to finalize the renaming in both paths
#                             for temp_folder in os.listdir(optical_scenario_path):
#                                 optical_temp_path = os.path.join(optical_scenario_path, temp_folder)
#                                 final_name = temp_folder.replace("_temp", "")
#                                 optical_final_path = os.path.join(optical_scenario_path, final_name)
#                                 os.rename(optical_temp_path, optical_final_path)
                                
#                                 visual_temp_path = os.path.join(visual_scenario_path, temp_folder)
#                                 visual_final_path = os.path.join(visual_scenario_path, final_name)
#                                 os.rename(visual_temp_path, visual_final_path)


# base_dir = "./images_shuffle/"
# shuffle_pairs(base_dir)

import os
import random
import shutil

def shuffle_and_copy_pairs(source_dir, target_dir):
    # optical_window30과 visual_window30을 셔플한 후, target_dir로 복사합니다.
    optical_source_dir = os.path.join(source_dir, 'optical_window30')
    visual_source_dir = os.path.join(source_dir, 'visual_window30')
    
    optical_target_dir = os.path.join(target_dir, 'optical_window30')
    visual_target_dir = os.path.join(target_dir, 'visual_window30')
    
    # 만약 target_dir이 없다면 생성합니다.
    os.makedirs(optical_target_dir, exist_ok=True)
    os.makedirs(visual_target_dir, exist_ok=True)
    
    for person_id in os.listdir(optical_source_dir):
        optical_person_path = os.path.join(optical_source_dir, person_id)
        visual_person_path = os.path.join(visual_source_dir, person_id)
        
        optical_person_target_path = os.path.join(optical_target_dir, person_id)
        visual_person_target_path = os.path.join(visual_target_dir, person_id)
        
        if os.path.isdir(optical_person_path) and os.path.isdir(visual_person_path):
            os.makedirs(optical_person_target_path, exist_ok=True)
            os.makedirs(visual_person_target_path, exist_ok=True)
            
            for section in ['A', 'B', 'C']:
                optical_section_path = os.path.join(optical_person_path, section)
                visual_section_path = os.path.join(visual_person_path, section)
                
                optical_section_target_path = os.path.join(optical_person_target_path, section)
                visual_section_target_path = os.path.join(visual_person_target_path, section)
                
                if os.path.isdir(optical_section_path) and os.path.isdir(visual_section_path):
                    os.makedirs(optical_section_target_path, exist_ok=True)
                    os.makedirs(visual_section_target_path, exist_ok=True)
                    
                    for scenario in ['bump', 'corner']:
                        optical_scenario_path = os.path.join(optical_section_path, scenario)
                        visual_scenario_path = os.path.join(visual_section_path, scenario)
                        
                        optical_scenario_target_path = os.path.join(optical_section_target_path, scenario)
                        visual_scenario_target_path = os.path.join(visual_section_target_path, scenario)
                        
                        if os.path.isdir(optical_scenario_path) and os.path.isdir(visual_scenario_path):
                            os.makedirs(optical_scenario_target_path, exist_ok=True)
                            os.makedirs(visual_scenario_target_path, exist_ok=True)
                            
                            # Get all round folders (1, 2, 3, 4)
                            round_folders = [folder for folder in os.listdir(optical_scenario_path) if os.path.isdir(os.path.join(optical_scenario_path, folder))]
                            
                            # Shuffle the round folders in a consistent way for both optical and visual
                            shuffled_folders = round_folders[:]
                            random.shuffle(shuffled_folders)
                            
                            # Copy the folders to the target directory with the new shuffled names
                            for original_name, shuffled_name in zip(round_folders, shuffled_folders):
                                # Define source and target paths
                                optical_original_path = os.path.join(optical_scenario_path, original_name)
                                visual_original_path = os.path.join(visual_scenario_path, original_name)
                                
                                optical_target_path = os.path.join(optical_scenario_target_path, shuffled_name)
                                visual_target_path = os.path.join(visual_scenario_target_path, shuffled_name)
                                
                                # Copy the original folder to the new shuffled target path
                                shutil.copytree(optical_original_path, optical_target_path)
                                shutil.copytree(visual_original_path, visual_target_path)

# 원본 이미지가 위치한 폴더
source_dir = "./images/"
# 셔플된 이미지를 저장할 폴더
target_dir = "./images_shuffle/"

shuffle_and_copy_pairs(source_dir, target_dir)



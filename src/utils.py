import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))

import numpy as np
import random
import cv2
from sklearn.model_selection import train_test_split

def generate_train_test_set(product_path, test_ratio=0.2):
    good_path = os.path.join(product_path, 'train')
    defect_path = os.path.join(product_path, 'test')
    
    good_image_list = [os.path.abspath(os.path.join(good_path, image_path)) for image_path in os.listdir(good_path)]
    train_good_image_list, test_good_image_list, _, _ = train_test_split(good_image_list, [1]*len(good_image_list),
                                                                   test_size=test_ratio, random_state=205)
    
    all_train_defect_image_list = []
    all_test_defect_image_list = []
    for case in os.listdir(defect_path):
        case_path = os.path.abspath(os.path.join(defect_path, case))
        defect_image_list = [os.path.abspath(os.path.join(case_path, image_path)) for image_path in os.listdir(case_path)]
        train_defect_image_list, test_defect_image_list, _, _ = train_test_split(defect_image_list, [0]*len(defect_image_list),
                                                                   test_size=test_ratio, random_state=205)
        # print(len(train_defect_image_list), len(test_defect_image_list))
        all_train_defect_image_list += train_defect_image_list
        all_test_defect_image_list += test_defect_image_list
    # print(len(all_train_defect_image_list), len(all_test_defect_image_list))
    
    return train_good_image_list, test_good_image_list, all_train_defect_image_list, all_test_defect_image_list    


def generate_train_test_set_naive(product_path, test_ratio):
    good_path = os.path.join(product_path, 'train')
    defect_path = os.path.join(product_path, 'test')
    
    good_image_list = [os.path.abspath(os.path.join(good_path, image_path)) for image_path in os.listdir(good_path)]
    train_good_image_list, test_good_image_list, _, _ = train_test_split(good_image_list, [1]*len(good_image_list),
                                                                   test_size=test_ratio, random_state=205)

    all_defect_image_list = []
    for case in os.listdir(defect_path):
        case_path = os.path.abspath(os.path.join(defect_path, case))

        defect_image_list = [os.path.abspath(os.path.join(case_path, image_path)) for image_path in os.listdir(case_path)]
        all_defect_image_list += all_defect_image_list

    all_train_defect_image_list, all_test_defect_image_list, _, _ = train_test_split(all_defect_image_list, [0]*len(good_image_list),
                                                                   test_size=test_ratio, random_state=205)

    return train_good_image_list, test_good_image_list, all_train_defect_image_list, all_test_defect_image_list  

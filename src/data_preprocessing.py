import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))

from pathlib import Path
import shutil
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.decomposition import PCA
from sklearn import preprocessing
import albumentations as albumentations
from data_analysis import * 
from utils import *


def augment_data(data_path, augmented_data_path):
    if not os.path.exists(augmented_data_path):
        os.mkdir(augmented_data_path)

    products = os.listdir(data_path)
    number_of_product = len(products)

    generate_data = True

    # Augment pipeline:
    transform = albumentations.Compose([
        albumentations.Rotate(limit=5),
        albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    ])

    for product in products:
        print(product)
        product_path = os.path.join(data_path, product)
        target_product_path = os.path.join(augmented_data_path, product)
        if not os.path.exists(target_product_path):
            os.mkdir(target_product_path)
        
        train_path = os.path.join(product_path, "train")
        test_path = os.path.join(product_path, "test")

        new_test_path = os.path.join(target_product_path, 'test')
        if not os.path.exists(new_test_path):
            os.mkdir(new_test_path)
        for case in os.listdir(test_path):
            if case != 'good':
                case_path = os.path.join(test_path, case)
                new_case_path = os.path.join(new_test_path, case)
                if not os.path.exists(new_case_path):
                    os.mkdir(new_case_path)
                for image in os.listdir(case_path):
                    test_image = cv2.imread(os.path.join(case_path, image), cv2.COLOR_BGR2RGB)

                    # Transform each image 4 times, and save result each time
                    for i in range(4):
                        transformed_image = transform(image=test_image)['image']
                        transformed_image_name = image[:-4] + '_transformed_' + str(i) + '.png'
                        if generate_data:
                            cv2.imwrite(os.path.join(new_case_path, transformed_image_name), transformed_image)

        new_train_path = os.path.join(target_product_path, 'train')
        if not os.path.exists(new_train_path):
            os.mkdir(new_train_path)
        for image_path in os.listdir(train_path):
            shutil.copy(os.path.join(train_path, image_path), os.path.join(new_train_path, image_path))


def group_data_to_one_bad_case(data_path, grouped_data_path, move_image=True):
    if not os.path.exists(grouped_data_path):
        os.mkdir(grouped_data_path)


    products = os.listdir(data_path)
    for product in products:
        print(product)
        product_path = os.path.join(data_path, product) if augment_data else os.path.join(data_path, product)
        train_path = os.path.join(product_path, "train")
        test_path = os.path.join(product_path, "test")
        # print(os.path.exists(train_path), os.path.exists(test_path))
        
        dest_product_path = os.path.join(grouped_data_path, product)
        dest_good_path = os.path.join(dest_product_path, "train")
        dest_test_path = os.path.join(dest_product_path, "test")
        dest_defect_path = os.path.join(dest_test_path, "defect")
        for dest_path in [dest_product_path, dest_good_path, dest_test_path, dest_defect_path]:
            if not os.path.exists(dest_path):
                os.mkdir(dest_path)
            # else:
            #     print(dest_path)

        for image in os.listdir(train_path):
            image_path = os.path.join(train_path, image)
            new_image_path = os.path.join(dest_good_path, image)
            if move_image:
                shutil.copy(image_path, new_image_path)
        for case in os.listdir(test_path):
            if case == 'good':
                case_path = os.path.join(test_path, case)
                for image in os.listdir(case_path):
                    image_path = os.path.join(case_path, image)
                    if image in os.listdir(train_path):
                        image = image[:-4] + '_1.png'
                    new_image_path = os.path.join(dest_good_path, image)
                    if move_image:
                        shutil.copy(image_path, new_image_path)
            else:
                case_path = os.path.join(test_path, case)
                for image in os.listdir(case_path):
                    image_path = os.path.join(case_path, image)
                    new_image_name = case + '_' + image
                    new_image_path = os.path.join(dest_defect_path, new_image_name)
                    if move_image:
                        shutil.copy(image_path, new_image_path)


def group_data_to_multiple_bad_case(data_path, grouped_data_path, move_image=True):
    if not os.path.exists(grouped_data_path):
        os.mkdir(grouped_data_path)


    products = os.listdir(data_path)
    for product in products:
        print(product)
        product_path = os.path.join(data_path, product) if augment_data else os.path.join(data_path, product)
        train_path = os.path.join(product_path, "train")
        test_path = os.path.join(product_path, "test")
        # print(os.path.exists(train_path), os.path.exists(test_path))
        
        train_path = os.path.join(product_path, "train")
        test_path = os.path.join(product_path, "test")
        # print(os.path.exists(train_path), os.path.exists(test_path))
        
        dest_product_path = os.path.join(grouped_data_path, product)
        dest_good_path = os.path.join(dest_product_path, "train")
        dest_test_path = os.path.join(dest_product_path, "test")
        for dest_path in [dest_product_path, dest_good_path, dest_test_path]:
            if not os.path.exists(dest_path):
                os.mkdir(dest_path)
            # else:
            #     print(dest_path)

        for image in os.listdir(train_path):
            image_path = os.path.join(train_path, image)
            new_image_path = os.path.join(dest_good_path, image)
            if move_image:
                shutil.copy(image_path, new_image_path)
        for case in os.listdir(test_path):
            if case == 'good':
                case_path = os.path.join(test_path, case)
                for image in os.listdir(case_path):
                    image_path = os.path.join(case_path, image)
                    if image in os.listdir(train_path):
                        image = image[:-4] + '_1.png'
                    new_image_path = os.path.join(dest_good_path, image)
                    if move_image:
                        shutil.copy(image_path, new_image_path)
            else:
                case_path = os.path.join(test_path, case)
                dest_case_path = os.path.join(dest_test_path, case)
                if not os.path.exists(dest_case_path):
                    os.mkdir(dest_case_path)
                for image in os.listdir(case_path):
                    image_path = os.path.join(case_path, image)
                    new_image_name = case + '_' + image
                    new_image_path = os.path.join(dest_case_path, new_image_name)
                    if move_image:
                        shutil.copy(image_path, new_image_path)


def train_test_split(data_path, target_path):
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    for product in products:
        # Split from unaugmented-ungrouped data path for unnaive split
        product_path = os.path.join(data_path , product)

        train_dict, test_dict, model_image_dict = generate_datasets_dict(product_path)

        product_target_path = os.path.join(target_path, product)
        if not os.path.exists(product_target_path):
            os.mkdir(product_target_path)

        for i in range(2):
            write_dict = [train_dict, test_dict][i]
            file_name = product + "_" + ['train_dict', 'test_dict'][i] + '.json'
            file_path = os.path.join(product_target_path, file_name)
            with open(file_path, 'w') as file:
                json.dump(write_dict, file)


if __name__ == '__main__':
    root_path = os.getcwd()
    data_path = os.path.join(root_path, "data/original_data")
    products = os.listdir(data_path)
    number_of_product = len(products)

    augment_data = False
    # Augment the data
    if augment_data:
        print("Augmenting data")
        augmented_data_path = os.path.join(root_path, "data/augmented_data")
        augment_data(data_path, augmented_data_path)

    group_one_bad_case = False
    if group_one_bad_case:
        # Group data to one bad case 
        # Unaugmented data
        print("Group Unaugmented data to one bad case")
        unaugmented_one_bad_case_path = os.path.join(root_path, "data/unaugmented_grouped_data")
        group_data_to_one_bad_case(data_path, unaugmented_one_bad_case_path)

        # Augmented data
        print("Group Augmented data to one bad case")
        augmented_one_bad_case_path = os.path.join(root_path, "data/augmented_grouped_data")
        group_data_to_one_bad_case(data_path, augmented_one_bad_case_path)


    group_multi_bad_case = True
    if group_multi_bad_case:
    # Group data to 
        # Unaugmented data
        print("Group Unaugmented data to multiple bad case")
        unaugmented_multi_bad_case_path = os.path.join(root_path, "data/unaugmented_multi_class_grouped_data")
        group_data_to_multiple_bad_case(data_path, unaugmented_multi_bad_case_path)

        # Augmented data
        print("Group Augmented data to multiple bad case")
        augmented_multi_bad_case_path = os.path.join(root_path, "data/augmented_multi_class_grouped_data")
        group_data_to_multiple_bad_case(data_path, augmented_multi_bad_case_path)

    



    # Train test split for unaugmented data from multiple bad case data
    print("Train test split for unaugmented data from multiple bad case data")
    unaugmented_target_path = os.path.join(root_path, 'data', 'unaugmented_train_test_split')
    train_test_split(unaugmented_multi_bad_case_path, unaugmented_target_path)

    # Train test split for augmented data
    print("Train test split for Augmented data from multiple bad case data")
    augmented_target_path = os.path.join(root_path, 'data', 'augmented_train_test_split')
    train_test_split(augmented_multi_bad_case_path, augmented_target_path)
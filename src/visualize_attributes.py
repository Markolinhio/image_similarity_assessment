from pathlib import Path
from tqdm import tqdm
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.decomposition import PCA
from sklearn import preprocessing
import plotly.graph_objects as go

def all_image_shape(data_path):
    products = os.listdir(data_path)
    number_of_product = len(products)
    
    train_shape = {}
    test_shape = {}
    for product in products:
        #print(product)
        product_path = os.path.join(data_path, product)
        
        train_path = os.path.join(product_path, "train")
        test_path = os.path.join(product_path, "test")
        train_shapes = []
        test_shapes = []
        for image in os.listdir(train_path):
            train_image = cv2.imread(os.path.join(train_path, image), cv2.IMREAD_UNCHANGED)
            train_shapes.append(train_image.shape[:2])
        for case in os.listdir(test_path):
            case_path = os.path.join(test_path, case)
            for image in os.listdir(case_path):
                test_image = cv2.imread(os.path.join(case_path, image), cv2.IMREAD_UNCHANGED)
                test_shapes.append(test_image.shape[:2])
        train_shape[product] = train_shapes
        test_shape[product] = test_shapes

    all_shape = {}
    for product in os.listdir(data_path):
        shape_list = train_shape[product] + test_shape[product]
        all_shape[product] = np.unique(np.array(shape_list), axis=0)

    return all_shape


def embeddings(data_path):
    products = os.listdir(data_path)
    number_of_product = len(products)

    labels_dict = {}
    embedded_images_dict = {}
    n_components = 2
    pca = PCA(n_components)
    
    for product in products:
        #print(product)
        product_path = os.path.join(data_path, product)
        
        train_path = os.path.join(product_path, "train")
        test_path = os.path.join(product_path, "test")
        
        labels = []
        flattened_images = []
        for image in os.listdir(train_path):
            train_image = cv2.imread(os.path.join(train_path, image), cv2.IMREAD_UNCHANGED)
            resized_train_image = cv2.resize(train_image, (64,64))
            flattened_train_image = resized_train_image.ravel()
            flattened_images.append(flattened_train_image)
    
            labels.append(product + " good")
        for case in os.listdir(test_path):
            if case != 'good':
                case_path = os.path.join(test_path, case)
                for image in os.listdir(case_path):
                    test_image = cv2.imread(os.path.join(case_path, image), cv2.IMREAD_UNCHANGED)
                    resized_test_image = cv2.resize(test_image, (64,64))
                    flattened_test_image = resized_test_image.ravel()
                    flattened_images.append(flattened_test_image)
        
                    labels.append(product + " " + case)
    
        embeddings = pca.fit_transform(flattened_images)
        labels_dict[product] = labels
        embedded_images_dict[product] = embeddings

    return embedded_images_dict, labels_dict


def class_distribution(data_path):
    products = os.listdir(data_path)
    number_of_product = len(products)

    labels_dict = {}
    for product in products:
        #print(product)
        labels = {}
        product_path = os.path.join(data_path, product)
        
        train_path = os.path.join(product_path, "train")
        test_path = os.path.join(product_path, "test")
        
        labels['good'] = len(os.listdir(train_path))
        
        for case in os.listdir(test_path): 
            case_path = os.path.join(test_path, case)
            if case == 'good':
                labels[case] += len(os.listdir(case_path))
            else:
                labels[case] = len(os.listdir(case_path))
        
        labels_dict[product] = labels

    return labels_dict
    
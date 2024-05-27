import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# Detect largest contour in the image along with its bounding rectangle and convex hull. Used for field segmentation and misc task when needed
def detect_largest_contour(image, threshold=False):
    # Contour detection
    if len(image.shape) == 2 or image.shape[2] == 1:
        if threshold:
            _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL,
                                        method=cv2.CHAIN_APPROX_NONE)
            
        else:
            contours, _ = cv2.findContours(image=image, mode=cv2.RETR_EXTERNAL,
                                        method=cv2.CHAIN_APPROX_NONE)

    else:
        return "Image should be one channel"

    if len(contours) > 0:    
        # Take the largest contour
        contours = max(contours, key=cv2.contourArea)

        # Find the bounding box corresponding to the contour
        rect = np.int16(cv2.boundingRect(contours))

        # Find the convex hull corresponding to the bounding box
        convex_hull = cv2.convexHull(contours)

        return contours, rect, convex_hull
    else:
        return None, None, None
    

def distance(vector_1, vector_2):
    #if vector_1.shape != vector_2.shape:
        #return None
    squared_diff = np.square(np.array([vector_1[i] - vector_2[i] for i in range(len(vector_1))]))
    distance = np.sqrt(np.sum(squared_diff))
    return distance


def cosine_similarity(vector_1, vector_2):
    #if vector_1.shape != vector_2.shape:
        #return None
    return np.dot(vector_1, vector_2)/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2))


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


def generate_train_test_set_naive(product_path, test_ratio=0.2):
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


def generate_datasets_dict(product_path, naive=False):
    if naive:
        train_good_image_list, test_good_image_list, all_train_defect_image_list, all_test_defect_image_list = generate_train_test_set_naive(product_path)
    else:
        train_good_image_list, test_good_image_list, all_train_defect_image_list, all_test_defect_image_list = generate_train_test_set(product_path)

    train_dict = dict(zip(train_good_image_list + all_train_defect_image_list, [1]*len(train_good_image_list) + [0]*len(all_train_defect_image_list)))
    test_dict = dict(zip(test_good_image_list + all_test_defect_image_list, [1]*len(test_good_image_list) + [0]*len(all_test_defect_image_list)))
    test_image_path = test_good_image_list[0]
    model_image_dict = dict(zip([test_image_path], [1]))

    return train_dict, test_dict, model_image_dict


class EmbeddingDataset(Dataset):
    def __init__(self, path_dict, transform=None):
        self.path_dict = path_dict
        self.transform = transform
        
    def __getitem__(self, index):
        image_path, label = list(self.path_dict.items())[index]
        image_1 = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if self.transform is not None:
            image_1 = self.transform(image_1)
        
        return image_1, label
    
    def __len__(self):
        return len(self.path_dict.keys())


def map_good_train_samples_to_embeddings(json_path, product_dict, save=False):
    # Filter only good samples
    good_images_dict = dict([(image_path, label) for image_path, label in product_dict.items()
                            if label == 1])

    # Remake image dict to key are image paths and values are the ids
    good_images_dict = dict(zip(list(good_images_dict.keys()), 
                                list(range(1, len(list(good_images_dict.keys())) + 1))))

    # Load ResNet50 model
    model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
    # model.fc = nn.Linear(2048, 2)
    modules=list(model.children())[:-1]
    model=nn.Sequential(*modules)
    for p in model.parameters():
        p.requires_grad = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Load data to model
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((1024,1024), antialias=True)])
    

    dataset = EmbeddingDataset(good_images_dict, transform)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True)
    
    # Calculate embeddings of all good samples
    all_labels = []
    all_embeddings = []
    torch.cuda.empty_cache()
    model.eval()
    for data,labels in tqdm(dataloader):
        new_labels = labels.numpy().tolist()
        all_labels += new_labels
        data = data.to(device)
        embeddings = model(data)
        all_embeddings.append(np.reshape(embeddings.detach().cpu().numpy(),(len(new_labels),-1)))
    all_embeddings = np.vstack(all_embeddings)

    # Save the embeddings and their ids
    if save:
        score_path = os.path.join(json_path, 'good_embeddings.csv')
        np.savetxt(score_path, all_embeddings, delimiter=",")
        np.savetxt(score_path[:-4] + '_id.csv',  all_labels, delimiter=",")

    return all_embeddings, all_labels
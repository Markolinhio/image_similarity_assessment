import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))

from pathlib import Path
import shutil
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_similarity_measures import quality_metrics as qm
from tqdm import tqdm
from utils import *
import time

def table_of_score(model_image_list, image_dict):
    score_list = []
    metrics = ['psnr', 'fsim', 'ssim', 'uiq', 'sam', 'sre'][3:] # remove 'rmse' value because other metrics are based on it
    for i in tqdm(range(len(image_dict.items()))):
        image_path, label = list(image_dict.items())[i]
        #start = time.time()
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Load model image
        model_image_path = model_image_list[i]
        model_label = 1
        model_image = cv2.imread(model_image_path, cv2.IMREAD_COLOR)

        scores = []
        for metric in metrics:
            try:
                metric_function = qm.metric_functions[metric]
                score = metric_function(model_image, image)
                scores.append(score)
            except:
                scores.append(np.nan)
        
        if model_label != label:
            scores.append(0.)
        else:
            scores.append(1.)
            
        score_list.append(np.array(scores))
        #end = time.time()
        #print("Time elapsed:", end - start,"sec\n")
    score_list = np.array(score_list)
    return score_list

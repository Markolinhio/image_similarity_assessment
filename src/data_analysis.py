from pathlib import Path
from tqdm import tqdm
import cv2
import os
import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
import plotly.graph_objects as go
from siamese import *
from utils import *

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import colormaps as cm
from matplotlib.gridspec import GridSpec

# Set plot parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 5.0
plt.rcParams['xtick.minor.size'] = 3.0
plt.rcParams['ytick.major.size'] = 5.0
plt.rcParams['ytick.minor.size'] = 3.0
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['legend.handlelength'] = 2.0
minor_locator = AutoMinorLocator(4)

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


def embeddings(data_path, embedder='pca', resnet_embedding=True):
    products = os.listdir(data_path)
    number_of_product = len(products)

    labels_dict = {}
    embedded_images_dict = {}
    n_components = 2
    if embedder == 'pca':
        embedder = PCA(n_components)
    elif embedder == 'tsne':
        embedder = TSNE(n_components)
    else:
        return None, None
    if resnet_embedding:
        model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
        # model.fc = nn.Linear(2048, 2)
        modules=list(model.children())[:-1]
        model=nn.Sequential(*modules)
        for p in model.parameters():
            p.requires_grad = False
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        label_case_dict = { 0 : 'good'}
        for product in products:
            product_path = os.path.join(data_path, product)
            cases_dict = {}
            for class_path in os.listdir(product_path):
                if class_path == 'test':
                    for case_name in os.listdir(os.path.join(product_path, class_path)):
                        case_code = os.listdir(os.path.join(product_path, class_path)).index(case_name) + 1
                        if case_code not in label_case_dict:
                            label_case_dict[case_code] = case_name
                        for image_path in os.listdir(os.path.join(product_path, class_path, case_name)):
                            cases_dict[os.path.join(product_path, class_path, case_name, image_path)] = case_code
                else:
                    for image_path in os.listdir(os.path.join(product_path, class_path)):
                        #id = os.listdir(os.path.join(product_path, class_path)).index(image_path)
                        cases_dict[os.path.join(product_path, class_path, image_path)] = 0
            
            transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((1024,1024), antialias=True)
                                    ])
            dataset = EmbeddingDataset(cases_dict, transform)
            dataloader = DataLoader(dataset, batch_size=12, shuffle=True)
            
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

            embeddings = embedder.fit_transform(all_embeddings)
            all_labels_name = [label_case_dict[label] for label in all_labels]
            labels_dict[product] = all_labels#all_labels_name
            embedded_images_dict[product] = embeddings 
            score_path = os.path.join("/media/khoa-ys/Personal/Materials/Master's Thesis/image_similarity_assessment", 'result/embeddings/' + product + '.csv')
            np.savetxt(score_path, all_embeddings, delimiter=",")
            np.savetxt(score_path[:-4] + '_id.csv',  all_labels, delimiter=",")
    else:
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
            flattened_images = np.array(flattened_images)
            embeddings = embedder.fit_transform(flattened_images)
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


def draw_class_distribution_histogram(data_path):
    root_path = os.getcwd()
    products = os.listdir(data_path)
    number_of_product = len(products)
    labels_dict = class_distribution(data_path)

    # Get good and defect counts of each product
    good_list_count = []
    defect_list_count = []
    for i in range(number_of_product):
        product = products[i]
        product_count = labels_dict[product]

        good_count = 0
        defects_count = 0
        for label, count in product_count.items():
            if label == 'good':
                good_count += count
            else:
                defects_count += count
        good_list_count.append(good_count)
        defect_list_count.append(defects_count)

    #print(good_list_count, defect_list_count)
    plt.figure(figsize=(24, 12), layout='tight')
    plt.bar(products, good_list_count, color='r')
    plt.bar(products, defect_list_count, bottom=good_list_count, color='b')
    plt.xlabel('Products', fontsize=22)
    plt.ylabel('Number of samples', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=18)
    plt.legend(["Good", "Defects"])
    plt.title("Good/Defects distributions among each product", fontsize=24)

    plt.savefig(root_path + '/Figure/stacked_bar_class_distribution.png')
    plt.show()


def data_analysis(root_path, data_path, check_shape=True, check_embeddings=True, check_dist=True, image_prefix='original_data'):
    products = os.listdir(data_path)
    number_of_product = len(products)
    # Check image shape of all types of product
    if check_shape:
        all_shape = all_image_shape(data_path)
        print(*all_shape.items(), sep="\n")

    # Check image embeddings
    if check_embeddings:
        embedder = 'tsne'
        embedded_images_dict, labels_dict = embeddings(data_path, embedder=embedder)
        
        label_encoder = preprocessing.LabelEncoder()
        for i in range(number_of_product):
            fig, ax = plt.subplots()
            plt.rcParams['axes.labelsize'] = 10
            plt.rcParams['axes.titlesize'] = 14
            product = products[i]
        
            embedded_images = embedded_images_dict[product]
            labels = labels_dict[product]
            labels_number = label_encoder.fit_transform(labels)
            
            scatter = ax.scatter(embedded_images[:, 0], embedded_images[:, 1], c=labels_number)
            ax.title.set_text(product)
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.legend(handles=scatter.legend_elements()[0], labels=np.unique(labels).tolist(), loc='upper right')
            fig.show()
            # fig.savefig(root_path + '/Figure/embeddings/' + product + '_' + image_prefix + '_each_class_embeddings_' + embedder + '.png')

    # Check class distribution
    if check_dist:
        labels_dict = class_distribution(data_path)
    
        # fig, ax = plt.subplots(5, 3, figsize=(30, 50))
        # fig.tight_layout(pad=5)
        # plt.rcParams['axes.labelsize'] = 18
        # plt.rcParams['axes.titlesize'] = 20

        good_list_count = []
        defect_list_count = []
        for i in range(number_of_product):
            product = products[i]
            product_count = labels_dict[product]

            good_count = 0
            defects_count = 0
            for label, count in product_count.items():
                if label == 'good':
                    good_count += count
                else:
                    defects_count += count
            good_list_count.append(good_count)
            defect_list_count.append(defects_count)

        print(good_list_count, defect_list_count)
        plt.figure(figsize=(24, 12), layout='tight')
        plt.bar(products, good_list_count, color='r')
        plt.bar(products, defect_list_count, bottom=good_list_count, color='b')
        plt.xlabel('Products', fontsize=22)
        plt.ylabel('Number of samples', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=18)
        plt.legend(["Good", "Defects"])
        plt.title("Good/Defects distributions among each product", fontsize=24)

            # labels, count = list(product_count.keys()), list(product_count.values())
            # print(labels, count)
            # x, y = (int(i%3), int(i/3))
            # ax[y, x].pie(count, labels=labels, autopct=lambda x: np.round(x/100.*np.sum(count), 0))
            # ax[y, x].title.set_text('Class distribution of '+ product)
            # ax[y, x].legend()
        plt.savefig(root_path + '/Figure/' + image_prefix + '_stacked_bar_distribution.png')
        plt.show()


def draw_embeddings_and_each_sample_case(product_path, embedder='TSNE', n_components=2, device='cuda'):
    if embedder.lower() == 'pca':
        embedder = PCA(n_components)
    elif embedder.lower() == 'tsne':
        embedder = TSNE(n_components)

    product = os.path.basename(product_path)
    good_path = os.path.join(product_path, 'train')
    all_cases_path = [good_path] + [os.path.join(product_path, 'test', case)
                                    for case in os.listdir(os.path.join(product_path, 'test'))]
    case_name_list = [os.path.basename(case_path) for case_path in all_cases_path]

    # Calculate embeddings
    # Generate class id dict for visualization
    label_case_dict = dict(zip(list(range(len(all_cases_path))), case_name_list))
    label_case_dict[0] = 'good'
    print(label_case_dict)

    # Load data, encode the labels to int and return the dict of {image_path:label_code}
    cases_dict = {}
    for case_path in all_cases_path:
        case_name = os.path.basename(case_path) if os.path.basename(case_path) != 'train' else 'good'
        case_code = [key for key, value in label_case_dict.items() if value == case_name][0]
        for image_path in os.listdir(case_path):
            cases_dict[os.path.join(case_path, image_path)] = case_code

    #print(*list(cases_dict.items()), sep='\n')
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((1024,1024), antialias=True)
                                    ])
    dataset = EmbeddingDataset(cases_dict, transform)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True)

    # Load ResNet model
    model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
    # model.fc = nn.Linear(2048, 2)
    modules=list(model.children())[:-1]
    model=nn.Sequential(*modules)
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)

    # Embed data
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

    # Put the embedded data through TSNE
    embeddings = embedder.fit_transform(all_embeddings)

    # Turn the string code
    all_labels_name = [label_case_dict[label] for label in all_labels]

    # Draw each image from each case and the overall embedding shape
    # Define plot parameters
    n_plots = len(case_name_list)
    n_col = 5
    n_row = (int(n_plots / (n_col - 2)) + (n_plots % (n_col - 2) > 0)) if (int(n_plots / (n_col - 2)) + (n_plots % (n_col - 2) > 0)) > 2 else 2
    
    #print(n_row, n_col)
    fig = plt.figure(layout="constrained", figsize=(16+6, n_row*4))
    
    # Create the structure: embedding plot on the left, samples on the right
    gs = GridSpec(n_row, 5, figure=fig)
    ax = fig.add_subplot(gs[:2, :2])
    
    # Draw embedding plot
    # Color code for the plot:
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'brown', 'pink', 'navy']
    # Draw scatter plot
    for label in label_case_dict.values():
        i = np.where(np.array(all_labels_name) == label)[0]
        embeddings_in_group = embeddings[i,:]
        ax.scatter(embeddings_in_group[:, 0], embeddings_in_group[:, 1], 
                    c=colors[list(label_case_dict.values()).index(label)], 
                    label=label)
    #scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=all_labels, cmap=color_map)
    ax.set_title('Embedding', fontsize=22)
    ax.legend()
    
    # Draw the sample plot
    for i in range(n_plots):
        # Load first image from each case:
        image = cv2.imread(os.path.join(all_cases_path[i], os.listdir(all_cases_path[i])[0]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        case_name = case_name_list[i] if case_name_list[i] != 'train' else 'good'
    
        # Draw images on remaining grid square
        # print(i // (n_col-2), 2 + i % (n_col-2))
        ax = fig.add_subplot(gs[i // (n_col-2), 2 + i % (n_col-2)])
        ax.imshow(image)
        ax.set_title(case_name, fontsize=22)
        ax.axis('off')

    # Save plot as files
    root_path = os.getcwd()
    fig.suptitle('Embeddings and each good and defect cases of: ' + product, fontsize=24, fontweight='bold')
    fig.savefig(root_path + '/Figure/data_analysis/' + product + '.png')


if __name__ == '__main__':
    root_path = os.getcwd()
    data_path = os.path.join(root_path, "data/multi_classed_grouped_data")
    products = os.listdir(data_path)
    number_of_product = len(products)

    # Generate class stack bar plots:
    draw_class_distribution_histogram(data_path)

    # Generate embedding images:
    for product in products:
        product_path = os.path.join(data_path, product)
        draw_embeddings_and_each_sample_case(product_path, device='cpu')
    
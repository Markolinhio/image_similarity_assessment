from utils import *

import time
from tqdm import tqdm
from wide_resnet import Wide_ResNet

from sklearn.model_selection import train_test_split


# class ProductDataset(Dataset):
#     def __init__(self, image_label_dict, transform=None):
#         self.image_label_dict = image_label_dict
#         self.good_image_label_dict = dict(
#             [
#                 (image_path, label)
#                 for image_path, label in image_label_dict.items()
#                 if label == 1
#             ]
#         )
#         # print(len(list(self.good_image_label_dict.keys())))
#         self.bad_image_label_dict = dict(
#             [
#                 (image_path, label)
#                 for image_path, label in image_label_dict.items()
#                 if label == 0
#             ]
#         )
#         # print(len(list(self.bad_image_label_dict.keys())))
#         self.all_good_combination = [
#             (image_path_1, image_path_2)
#             for image_path_1 in self.good_image_label_dict.keys()
#             for image_path_2 in self.good_image_label_dict.keys()
#             if image_path_2 != image_path_1
#         ]
#         # print(len(self.all_good_combination))
#         self.all_bad_combination = [
#             (image_path_1, image_path_2)
#             for image_path_1 in self.good_image_label_dict.keys()
#             for image_path_2 in self.bad_image_label_dict.keys()
#         ]
#         # print(len(self.all_bad_combination))
#         self.all_combination = self.all_good_combination + self.all_bad_combination

#         # print(len(self.all_combination))
#         self.transform = transform

#     def __getitem__(self, index):
#         if len(list(self.good_image_label_dict.keys())) == 1:
#             # print(self.image_label_dict)
#             image_path_1 = list(self.good_image_label_dict.keys())[0]
#             image_1 = cv2.imread(image_path_1, cv2.IMREAD_COLOR)
#             label_1 = self.good_image_label_dict[image_path_1]
#             if self.transform is not None:
#                 image_1 = self.transform(image_1)
#             return (
#                 image_1,
#                 torch.empty((0, 0), dtype=torch.float32),
#                 torch.from_numpy(np.array([label_1])),
#                 torch.empty((0, 0), dtype=torch.float32),
#             )
#         else:
#             # We need to approximately 50% of images to be in the same class
#             # should_get_same_class = random.randint(0,1)
#             image_path_1, image_path_2 = self.all_combination[index]
#             label_1 = self.image_label_dict[image_path_1]
#             label_2 = self.image_label_dict[image_path_2]

#             image_1 = cv2.imread(image_path_1, cv2.IMREAD_COLOR)
#             image_2 = cv2.imread(image_path_2, cv2.IMREAD_COLOR)

#             if self.transform is not None:
#                 image_1 = self.transform(image_1)
#                 image_2 = self.transform(image_2)

#             return (
#                 image_1,
#                 image_2,
#                 torch.from_numpy(np.array([label_1, label_2])),
#                 torch.from_numpy(np.array([int(label_1 != label_2)], dtype=np.float32)),
#             )

#     def __len__(self):
#         return len(self.all_combination)


class ProductDataset(Dataset):
    def __init__(self, image_label_dict, transform=None):
        self.image_label_dict = image_label_dict
        self.good_image_label_dict = dict(
            [
                (image_path, label)
                for image_path, label in image_label_dict.items()
                if label == 1
            ]
        )
        self.transform = transform

    def __getitem__(self, index):
        if len(list(self.image_label_dict.keys())) == 1:
            # print(self.image_label_dict)
            image_path_1 = list(self.image_label_dict.keys())[0]
            image_1 = cv2.imread(image_path_1, cv2.IMREAD_COLOR)
            label_1 = self.image_label_dict[image_path_1]
            if self.transform is not None:
                image_1 = self.transform(image_1)
            return (
                image_1,
                torch.empty((0, 0), dtype=torch.float32),
                torch.from_numpy(np.array([label_1])),
                torch.empty((0, 0), dtype=torch.float32),
            )
        else:
            # print(len(self.good_image_label_dict.keys()))
            # We need to approximately 50% of images to be in the same class
            should_get_same_class = random.randint(0, 1)
            image_path_1 = list(self.good_image_label_dict.keys())[index]
            label_1 = self.image_label_dict[image_path_1]
            if should_get_same_class:
                same_label_list = [
                    image_path
                    for image_path, label in self.image_label_dict.items()
                    if image_path != image_path_1 and label == label_1
                ]
                image_path_2 = random.choice(same_label_list)
                label_2 = self.image_label_dict[image_path_2]
            else:
                opposite_label_list = [
                    image_path
                    for image_path, label in self.image_label_dict.items()
                    if image_path != image_path_1 and label != label_1
                ]
                image_path_2 = random.choice(opposite_label_list)
                label_2 = self.image_label_dict[image_path_2]

            image_1 = cv2.imread(image_path_1, cv2.IMREAD_COLOR)
            image_2 = cv2.imread(image_path_2, cv2.IMREAD_COLOR)

            if self.transform is not None:
                image_1 = self.transform(image_1)
                image_2 = self.transform(image_2)

            return (
                image_1,
                image_2,
                torch.from_numpy(np.array([label_1, label_2])),
                torch.from_numpy(np.array([int(label_1 != label_2)], dtype=np.float32)),
            )

    def __len__(self):
        if len(list(self.image_label_dict.keys())) == 1:
            return 1
        else:
            return len(self.good_image_label_dict.keys())


class SiameseNetwork(nn.Module):

    def __init__(self, backbone):
        super(SiameseNetwork, self).__init__()
        self.backbone = backbone
        if "wide" not in backbone:
            if "resnet" in backbone:
                if backbone == "resnet18":
                    # Setting up model with resnet18 backbone
                    backbone = torchvision.models.resnet18()
                    backbone.fc = nn.Linear(512, 1024)
                elif backbone == "resnet34":
                    # Setting up model with resnet34 backbone
                    backbone = torchvision.models.resnet34()

                    backbone.fc = nn.Linear(512, 1024)
                elif backbone == "resnet50":
                    # Setting up model with resnet34 backbone
                    backbone = torchvision.models.resnet50()
                    backbone.fc = nn.Linear(2048, 1024)
            elif backbone == "vgg":
                backbone = torchvision.models.vgg11()
                backbone.classifier[6] = nn.Linear(4096, 1024)
                # Make a
            self.cnn = nn.Sequential(
                backbone,
                nn.ReLU(True),
                nn.Linear(1024, 2048),
                nn.ReLU(True),
                nn.Linear(2048, 2048),
                nn.ReLU(True),
                nn.Linear(2048, 256),
                nn.ReLU(True),
                nn.Linear(256, 2),
            )
        else:
            self.cnn = Wide_ResNet(16, 4, 0.1, 2)

    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        output = self.cnn(x)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate the euclidean distance and calculate the contrastive loss
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive


def model_train(
    model,
    train_dataloader,
    epochs=500,
    lr=1e-4,
    batch_size=4,
    step_size=50,
    device="cpu",
    visualize=True,
    figure_path=None,
    save_model=True,
    model_path=None,
    resume=False,
    resume_model_path=None,
):
    # print(device)
    # print(figure_path)
    model = model.to(device)
    if resume and resume_model_path is not None:
        model.load_state_dict(torch.load(resume_model_path))
    loss = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

    torch.cuda.empty_cache()

    loss_history = []

    # Iterate throught the epochs
    for epoch in tqdm(range(epochs)):

        # print("------------ Epoch:", epoch, "------------")
        # Iterate over batches
        epoch_loss = 0
        iter_number = 0
        for i, (image_1, image_2, class_label, label) in enumerate(train_dataloader, 0):

            # Send the images and labels to CUDA
            image_1, image_2, label = (
                image_1.to(device),
                image_2.to(device),
                label.to(device),
            )

            # Zero the gradients
            optimizer.zero_grad()

            # Pass in the two images into the network and obtain two outputs
            output_1, output_2 = model(image_1, image_2)

            # Pass the outputs of the networks and label into the loss function
            loss_contrastive = loss(output_1, output_2, label)

            # Calculate the backpropagation
            loss_contrastive.backward()

            # Optimize
            optimizer.step()

            # Every 10 batches print out the loss
            iter_loss = loss_contrastive.item()
            epoch_loss += iter_loss
            iter_number += 1

            if save_model and model_path is not None and epoch % step_size == 0:
                temp_model_path = model_path[:-4] + "_" + str(epoch) + ".pth"
                torch.save(model.state_dict(), temp_model_path)
        avg_loss = epoch_loss / iter_number
        # print(f"Average loss {avg_loss}")

        loss_history.append(avg_loss)
        scheduler.step()

    fig, ax = plt.subplots(2, 1, figsize=(20, 10))
    ax[0].plot(range(1, epochs + 1), loss_history)
    ax[0].set_title("Average loss per epoch")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[1].plot(range(1, epochs + 1), loss_history)
    ax[1].set_yscale("log")
    ax[1].set_title("Average loss per epoch in log scale")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    if visualize:
        fig.show()

    if figure_path is not None:
        fig.savefig(figure_path)
    torch.cuda.empty_cache()

    if save_model and model_path is not None:
        torch.save(model.state_dict(), model_path)

    return model


def inference(
    model,
    model_dataloader,
    test_dataloader,
    product,
    device="cpu",
    visualize=True,
    score_path=None,
    figure_path=None,
):
    model = model.to(device)
    # Extract one batch
    model_image, _, model_label, _ = next(iter(model_dataloader))

    result = []
    for i in range(len(test_dataloader)):
        _, test_image, test_label, _ = next(iter(test_dataloader))

        output_1, output_2 = model(model_image.to(device), test_image.to(device))
        euclidean_distance = F.pairwise_distance(output_1, output_2, keepdim=True)

        image_1 = np.array(model_image[0].permute(1, 2, 0) * 255, dtype=np.uint8)
        test_image = np.array(test_image[0].permute(1, 2, 0) * 255, dtype=np.uint8)
        label_1 = "good"
        label_2 = "defect" if test_label.numpy()[0][1] == 0 else "good"
        label = "same class" if label_1 == label_2 else "different class"

        if visualize and i <= 4:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(image_1[:, :, ::-1])
            ax[0].set_xlabel(label_1, weight="bold", fontsize=20)
            # ax[0].axis('off')
            ax[1].imshow(test_image[:, :, ::-1])
            ax[1].set_xlabel(label_2, weight="bold", fontsize=20)
            # ax[1].axis('off')
            fig.suptitle("Dissimilarity score: " + str(euclidean_distance.item()))
            # fig.show()
        result.append(
            np.array(
                [
                    1.0,
                    test_label.numpy()[0][1],
                    1.0 == test_label.numpy()[0][1],
                    euclidean_distance.item(),
                ]
            )
        )
        torch.cuda.empty_cache()
    result = np.array(result)

    if score_path is not None:
        np.savetxt(score_path, result, delimiter=",")

    same_class = result[:, 2]
    similarity_scores = result[:, 3]

    dissimillar_idx = np.where(same_class == 0)
    dissimillar_score = similarity_scores[dissimillar_idx]
    simillar_idx = np.where(same_class == 1)
    simillar_score = similarity_scores[simillar_idx]
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].hist(
        simillar_score,
        bins=np.arange(simillar_score.min(), simillar_score.max(), 0.1),
        edgecolor="black",
    )
    ax[0].set_xlabel("Distance", fontsize=10)
    ax[0].set_ylabel("Number of instance", fontsize=10)
    ax[0].set_title(
        product + ": Similarity score distribution for images with same class"
    )
    ax[1].hist(
        dissimillar_score,
        bins=np.arange(dissimillar_score.min(), dissimillar_score.max(), 0.1),
        edgecolor="black",
    )
    ax[1].set_xlabel("Distance", fontsize=10)
    ax[1].set_ylabel("Number of instance", fontsize=10)
    ax[1].set_title(
        product + ": Similarity score distribution for images with different class"
    )
    if visualize:
        fig.show()

    if figure_path is not None:
        fig.savefig(figure_path)

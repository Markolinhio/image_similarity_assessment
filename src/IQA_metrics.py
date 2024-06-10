from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import torchvision
import torch

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    RocCurveDisplay,
    roc_auc_score,
    f1_score,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier


import time
from utils import *
from image_similarity_measures import quality_metrics as qm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import shutil
from pathlib import Path
import os


def table_of_score(model_image_list, image_dict):
    score_list = []
    # remove 'rmse' value because other metrics are based on it
    metrics = ['psnr', 'fsim', 'ssim', 'uiq', 'sam', 'sre'][3:]
    for i in tqdm(range(len(image_dict.items()))):
        image_path, label = list(image_dict.items())[i]
        # start = time.time()
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
        # end = time.time()
        # print("Time elapsed:", end - start,"sec\n")
    score_list = np.array(score_list)
    return score_list


def correlation_matrix(df, figure_path):
    corr_mat = df.corr()
    plt.figure(layout="constrained")
    sns.heatmap(corr_mat, annot=True)
    plt.savefig(figure_path)

    return corr_mat


class TableofScoreDataset(Dataset):
    def __init__(self, data, data_label):
        self.data = data
        self.data_label = data_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_i = self.data[idx]
        data_label_i = self.data_label[idx]

        return data_i, data_label_i


class SimpleMLP(torch.nn.Module):

    def __init__(self, n_class):
        super(SimpleMLP, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3, 2**3),
            torch.nn.BatchNorm1d(2**3),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2**3, 2**8),
            torch.nn.BatchNorm1d(2**8),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2**8, 2**11),
            torch.nn.BatchNorm1d(2**11),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2**11, 2**8),
            torch.nn.BatchNorm1d(2**8),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2**8, 2**3),
            torch.nn.BatchNorm1d(2**3),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2**3, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


def fit_svm(
    X_train, X_test, y_train, y_test, kernel="rbf", C=1, gamma="auto", n_splits=5, figure_path=None, plot_off=True
):
    # Define stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    f1_scores = []
    roc_auc_scores = []

    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Train and predict
        # Train and predict
        svm = SVC(probability=True, kernel=kernel, C=C,
                  gamma=gamma, class_weight='balanced')
        svm.fit(X_train_fold, y_train_fold)
        y_val_pred = svm.predict(X_val_fold)
        y_val_pred_prob = svm.predict_proba(X_val_fold)[:, 1]

        # Calculate f1-score
        f1 = f1_score(y_val_fold, y_val_pred)
        f1_scores.append(f1)

        # Calculate ROC-AUC score
        fpr, tpr, _ = roc_curve(y_val_fold, y_val_pred_prob)
        roc_auc = auc(fpr, tpr)
        roc_auc_scores.append(roc_auc)

    # Calculate average cross-validated scores
    avg_f1 = np.mean(f1_scores)
    avg_roc_auc = np.mean(roc_auc_scores)

    # Train the final model on the entire training set
    # Train and predict
    svm = SVC(probability=True, kernel=kernel, C=C,
              gamma=gamma, class_weight='balanced')
    svm.fit(X_train, y_train)
    y_test_pred = svm.predict(X_test)
    y_test_pred_prob = svm.predict_proba(X_test)[:, 1]

    # Visualize confusion matrix for the test set
    plt.figure(layout="constrained")
    sns.heatmap(confusion_matrix(y_test, y_test_pred,
                labels=np.unique(y_test)), annot=True)
    if figure_path:
        plt.savefig(figure_path)
    if plot_off:
        plt.close()

    # Calculate final test set scores
    test_f1 = f1_score(y_test, y_test_pred)
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_prob)
    test_roc_auc = auc(fpr, tpr)

    return avg_f1, avg_roc_auc, test_f1, test_roc_auc


def fit_ridge(
    X_train, X_test, y_train, y_test, alpha=1.0, n_splits=5, figure_path=None, plot_off=True
):
    # Define stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    f1_scores = []
    roc_auc_scores = []

    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Train and predict
        ridge = RidgeClassifier(alpha=alpha, class_weight='balanced')

        ridge.fit(X_train_fold, y_train_fold)

        # Predict on validation set
        y_val_pred = ridge.predict(X_val_fold)

        # Calculate f1-score for the fold
        f1 = f1_score(y_val_fold, y_val_pred)
        f1_scores.append(f1)

        # Calculate ROC-AUC score for the fold (RidgeClassifier does not support predict_proba)
        roc_auc = roc_auc_score(y_val_fold, y_val_pred)
        roc_auc_scores.append(roc_auc)

    # Calculate average cross-validated scores
    avg_f1 = np.mean(f1_scores)
    avg_roc_auc = np.mean(roc_auc_scores)

    # Train the final model on the entire training set
    ridge = RidgeClassifier(alpha=alpha, class_weight='balanced')
    ridge.fit(X_train, y_train)

    # Predict on the test set
    y_test_pred = ridge.predict(X_test)

    # Visualize confusion matrix
    plt.figure(layout="constrained")
    sns.heatmap(confusion_matrix(y_test, y_test_pred,
                labels=np.unique(y_test)), annot=True)
    if figure_path:
        plt.savefig(figure_path)
    if plot_off:
        plt.close()

    # Calculate f1-score on the test set
    test_f1 = f1_score(y_test, y_test_pred)

    # Calculate ROC-AUC score on the test set
    test_roc_auc = roc_auc_score(y_test, y_test_pred)

    return avg_f1, avg_roc_auc, test_f1, test_roc_auc


def fit_rf(X_train, X_test, y_train, y_test, random_state=0, n_splits=5, figure_path=None, plot_off=True):

    # Define stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    f1_scores = []
    roc_auc_scores = []
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        # Train and predict
        rf = RandomForestClassifier(
            random_state=random_state, class_weight='balanced')

        # Train and predict
        rf.fit(X_train_fold, y_train_fold)
        y_val_pred = rf.predict(X_val_fold)
        y_val_pred_prob = rf.predict_proba(X_val_fold)[:, 1]

        # Calculate f1-score
        f1 = f1_score(y_val_fold, y_val_pred)
        f1_scores.append(f1)

        # Calculate ROC-AUC score
        fpr, tpr, _ = roc_curve(y_val_fold, y_val_pred_prob)
        roc_auc = auc(fpr, tpr)
        roc_auc_scores.append(roc_auc)

    # Calculate average cross-validated scores
    avg_f1 = np.mean(f1_scores)
    avg_roc_auc = np.mean(roc_auc_scores)

    # Train the final model on the entire training set
    # Train and predict
    rf = RandomForestClassifier(
        random_state=random_state, class_weight='balanced')
    rf.fit(X_train, y_train)
    y_test_pred = rf.predict(X_test)
    y_test_pred_prob = rf.predict_proba(X_test)[:, 1]

    # Visualize confusion matrix for the test set
    plt.figure(layout="constrained")
    sns.heatmap(confusion_matrix(y_test, y_test_pred,
                labels=np.unique(y_test)), annot=True)
    if figure_path:
        plt.savefig(figure_path)
    if plot_off:
        plt.close()

    # Calculate final test set scores
    test_f1 = f1_score(y_test, y_test_pred)
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_prob)
    test_roc_auc = auc(fpr, tpr)

    return avg_f1, avg_roc_auc, test_f1, test_roc_auc


def fit_xgb(X_train, X_test, y_train, y_test, random_state=0, n_splits=5, figure_path=None, plot_off=True):
    # Define stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)

    f1_scores = []
    roc_auc_scores = []

    # Perform cross-validation
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Train and predict
        xgb = XGBClassifier(random_state=random_state,
                            use_label_encoder=False, eval_metric='logloss')
        xgb.fit(X_train_fold, y_train_fold)
        y_val_pred = xgb.predict(X_val_fold)
        y_val_pred_prob = xgb.predict_proba(X_val_fold)[:, 1]

        # Calculate F1 score
        f1 = f1_score(y_val_fold, y_val_pred)
        f1_scores.append(f1)

        # Calculate ROC-AUC score
        fpr, tpr, _ = roc_curve(y_val_fold, y_val_pred_prob)
        roc_auc = auc(fpr, tpr)
        roc_auc_scores.append(roc_auc)

    # Calculate average cross-validated scores
    avg_f1 = np.mean(f1_scores)
    avg_roc_auc = np.mean(roc_auc_scores)

    # Train the final model on the entire training set
    xgb = XGBClassifier(random_state=random_state,
                        use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    y_test_pred = xgb.predict(X_test)
    y_test_pred_prob = xgb.predict_proba(X_test)[:, 1]

    # Visualize confusion matrix for the test set
    plt.figure(layout="constrained")
    sns.heatmap(confusion_matrix(y_test, y_test_pred,
                labels=np.unique(y_test)), annot=True)
    if figure_path:
        plt.savefig(figure_path)
    if plot_off:
        plt.close()

    # Calculate final test set scores
    test_f1 = f1_score(y_test, y_test_pred)
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_prob)
    test_roc_auc = auc(fpr, tpr)

    return avg_f1, avg_roc_auc, test_f1, test_roc_auc


def fit_logit(X_train, X_test, y_train, y_test, C=1, random_state=0, n_splits=5, figure_path=None, plot_off=True):

    # Define stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    f1_scores = []
    roc_auc_scores = []
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Train and predict
        logit = LogisticRegression(C=C,
                                   random_state=random_state, class_weight='balanced')
        logit.fit(X_train_fold, y_train_fold)
        y_val_pred = logit.predict(X_val_fold)
        y_val_pred_prob = logit.predict_proba(X_val_fold)[:, 1]

        # Calculate f1-score
        f1 = f1_score(y_val_fold, y_val_pred)
        f1_scores.append(f1)

        # Calculate ROC-AUC score
        fpr, tpr, _ = roc_curve(y_val_fold, y_val_pred_prob)
        roc_auc = auc(fpr, tpr)
        roc_auc_scores.append(roc_auc)

    # Calculate average cross-validated scores
    avg_f1 = np.mean(f1_scores)
    avg_roc_auc = np.mean(roc_auc_scores)

    # Train the final model on the entire training set
    # Train and predict
    logit = LogisticRegression(C=C,
                               random_state=random_state, class_weight='balanced')
    logit.fit(X_train, y_train)
    y_test_pred = logit.predict(X_test)
    y_test_pred_prob = logit.predict_proba(X_test)[:, 1]

    # Visualize confusion matrix for the test set
    plt.figure(layout="constrained")
    sns.heatmap(confusion_matrix(y_test, y_test_pred,
                labels=np.unique(y_test)), annot=True)
    if figure_path:
        plt.savefig(figure_path)
    if plot_off:
        plt.close()

    # Calculate final test set scores
    test_f1 = f1_score(y_test, y_test_pred)
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_prob)
    test_roc_auc = auc(fpr, tpr)

    return avg_f1, avg_roc_auc, test_f1, test_roc_auc


def fit_mlp(
    X_train,
    X_test,
    y_train,
    y_test,
    lr=5e-4,
    batch_size=16,
    epochs=400,
    n_splits=5,
    train_figure_path=None,
    figure_path=None,
    verbose=False,
    plot_off=True
):
    n_classes = len(np.unique(y_train))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    f1_scores = []
    roc_auc_scores = []

    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        if verbose:
            print(f"Fold {fold+1}/{n_splits}")
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Specify the MLP model
        model = SimpleMLP(n_classes).to(device).double()

        # Load train, validation sets to dataloader
        train_dataset = TableofScoreDataset(X_train_fold, y_train_fold)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TableofScoreDataset(X_val_fold, y_val_fold)
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True)

        # Identify loss function and optimizer
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=1e-4)

        # Train model
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=100, gamma=0.1)
        for epoch in range(epochs):
            model.train()
            training_loss = 0
            for train_X, train_y in train_dataloader:
                train_X, train_y = train_X.to(device), train_y.to(device)
                optimizer.zero_grad()
                pred = model(train_X)
                loss = loss_fn(pred, train_y.view(-1, 1).double())
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
            scheduler.step()
            if verbose:
                print(
                    f"Epoch: {epoch}, Training loss: {training_loss/len(train_dataloader):.6f}")

        # Evaluate on validation set
        model.eval()
        val_predictions, val_true_labels, val_pred_prob = [], [], []
        with torch.no_grad():
            for val_X, val_y in val_dataloader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                pred = (model(val_X) > 0.5).float()
                val_predictions += pred.cpu().numpy().tolist()
                val_true_labels += val_y.cpu().numpy().tolist()
                val_pred_prob += model(val_X).cpu().numpy().tolist()
        y_val, y_val_pred, val_pred_prob = np.array(
            val_true_labels), np.array(val_predictions), np.array(val_pred_prob)
        # print(y_val.shape, y_val_pred.shape, val_pred_prob.shape)
        # Calculate f1-score for the fold
        f1 = f1_score(y_val, y_val_pred)
        f1_scores.append(f1)

        # Calculate ROC-AUC score for the fold
        fpr, tpr, _ = roc_curve(y_val, val_pred_prob)
        roc_auc = auc(fpr, tpr)
        roc_auc_scores.append(roc_auc)

    # Calculate average cross-validated scores
    avg_f1 = np.mean(f1_scores)
    avg_roc_auc = np.mean(roc_auc_scores)

    # Train the final model on the entire training set
    model = SimpleMLP(n_classes).to(device).double()
    train_dataset = TableofScoreDataset(X_train, y_train)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TableofScoreDataset(X_test, y_test)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.1)
    for epoch in range(epochs):
        model.train()
        training_loss = 0
        for train_X, train_y in train_dataloader:
            train_X, train_y = train_X.to(device), train_y.to(device)
            optimizer.zero_grad()
            pred = model(train_X)
            loss = loss_fn(pred, train_y.view(-1, 1).double())
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        scheduler.step()
        if verbose:
            print(
                f"Epoch: {epoch}, Training loss: {training_loss/len(train_dataloader):.6f}")

    # Evaluate model on the test set
    model.eval()
    test_predictions, true_labels, test_pred_prob = [], [], []
    with torch.no_grad():
        for test_X, test_y in test_dataloader:
            test_X, test_y = test_X.to(device), test_y.to(device)
            pred = (model(test_X) > 0.5).float()
            test_predictions += pred.cpu().numpy().tolist()
            true_labels += test_y.cpu().numpy().tolist()
            test_pred_prob += model(test_X).cpu().numpy().tolist()
    y_test, y_test_pred, test_pred_prob = np.array(
        true_labels), np.array(test_predictions), np.array(test_pred_prob)

    # Visualize confusion matrix
    plt.figure(layout="constrained")
    sns.heatmap(confusion_matrix(true_labels, test_predictions,
                labels=range(0, n_classes)), annot=True)
    if figure_path:
        plt.savefig(figure_path)
    if plot_off:
        plt.close()

    # Calculate f1-score on the test set
    test_f1 = f1_score(y_test, y_test_pred)

    # Calculate ROC-AUC score on the test set
    fpr, tpr, _ = roc_curve(y_test, test_pred_prob)
    test_roc_auc = auc(fpr, tpr)

    return avg_f1, avg_roc_auc, test_f1, test_roc_auc


def drop_correlated(df, threshold=0.7):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(
        upper[column] > threshold)]
    df_reduced = df.drop(columns=to_drop)
    return df_reduced


# Class balance
def plot_class_balance(df, target_column, fig_path, test=False):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_column, data=df)
    plt.title('Class Balance')
    plt.xlabel('Class')
    plt.ylabel('Count')
    fig_name = 'class_balance.png' if not test else 'class_balance_test.png'
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.close()

# Distribution of each feature


def plot_feature_distributions(df, target_column, fig_path, test=False):
    for column in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x=column, hue=target_column, kde=True,
                     element="step", stat="density", common_norm=False)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        fig_name = f'distribution_{column}.png' if not test else f'distribution_{column}_test.png'
        plt.savefig(os.path.join(fig_path, fig_name))
        plt.close()
        plt.close("all")
# Check collinearity


def plot_collinearity(df, fig_path, test=False):
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Collinearity Matrix')
    fig_name = 'collinearity_matrix.png' if not test else 'collinearity_matrix_test.png'
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.close()


# Analyze feature's impact on the label using Mutual Information
def plot_feature_impact(df, target_column, fig_path, test=False):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    mutual_info = mutual_info_classif(X, y, discrete_features='auto')
    mi_series = pd.Series(
        mutual_info, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=mi_series.values, y=mi_series.index)
    plt.title('Feature Impact on the Label (Mutual Information)')
    plt.xlabel('Mutual Information Score')
    plt.ylabel('Features')
    fig_name = 'feature_impact.png' if not test else 'feature_impact_test.png'
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.close()


# Pairplot
def plot_pairplot(df, target_column, fig_path, test=False):
    plt.figure(figsize=(12, 10))
    sns.pairplot(df, hue=target_column)
    fig_name = 'pairplot.png' if not test else 'pairplot_test.png'
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.close()

# Boxplot


def plot_boxplot(df, target_column, fig_path, test=False):
    for column in df.columns:
        if column != target_column:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=target_column, y=column, data=df)
            plt.title(f'Boxplot of {column} by {target_column}')
            plt.xlabel(target_column)
            plt.ylabel(column)
            fig_name = f'boxplot_{column}.png' if not test else f'boxplot_{column}_test.png'
            plt.savefig(os.path.join(fig_path, fig_name))
            plt.savefig(os.path.join(fig_path, f'boxplot_{column}.png'))
            plt.close()
            plt.close("all")

# EDA pipeline


def eda_pipeline(df, target_column, fig_path, test=False):
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # Step 1: Class Balance
    plot_class_balance(df, target_column, fig_path, test)

    # Step 2: Distribution of Each Feature
    plot_feature_distributions(df, target_column, fig_path, test)

    # Step 3: Collinearity
    plot_collinearity(df, fig_path)

    # Step 4: Feature Impact on the Label
    plot_feature_impact(df, target_column, fig_path, test)

    # Additional EDA Step 1: Pairplot
    plot_pairplot(df, target_column, fig_path, test)

    # Additional EDA Step 2: Boxplot
    plot_boxplot(df, target_column, fig_path, test)

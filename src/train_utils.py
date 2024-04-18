import os.path

import numpy as np
import seaborn
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from loguru import logger

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    return accuracy
def eval_metrics(model, test_loader, classes):
    y_trues = None
    y_preds = None
    iter = False

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            labels_np = labels.numpy()
            pred_np = predicted.numpy()
            if not iter:
                y_trues = labels_np
                y_preds = pred_np
                iter = True
            else:
                y_trues = np.concatenate((y_trues, labels_np), axis=0)
                y_preds = np.concatenate((y_preds, pred_np), axis=0)

    # acc = metrics.accuracy_score(y_trues, y_preds)
    # precision = metrics.precision_score(y_trues, y_preds)
    # recall = metrics.recall_score(y_trues, y_preds, pos_label=1)
    # f1_score = metrics.f1_score(y_trues, y_preds)
    plot_cm(y_trues, y_preds, classes=classes)
    return metrics.classification_report(y_trues, y_preds, zero_division=0)
def plot_graphs(train_losses, test_accuracies):
    plt.figure(figsize=(12, 6))

    # training loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    # test accuracy 그래프
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_cm(y_true, y_pred, classes, show='True'):

    mapped_y_true = np.array(classes)[y_true]
    mapped_y_pred = np.array(classes)[y_pred]

    cm = confusion_matrix(mapped_y_true, mapped_y_pred, labels=classes)
    print(cm)
    cm_vis = seaborn.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()



def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    train_losses = []  # 각 epoch의 training loss 기록
    test_accuracies = []  # 각 epoch의 test accuracy 기록

    for epoch in range(num_epochs):
        model.train()  # 모델을 training 모드로 설정

        idx = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            idx += 1
            # 현재 epoch의 training loss 기록
            if idx == len(train_loader):
                train_losses.append(loss.item())


        # 현재 epoch의 test accuracy 계산 및 기록
        test_accuracy = evaluate_model(model, test_loader)
        test_accuracies.append(test_accuracy)

        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {test_accuracy:.2f}%')

    # training loss 및 test accuracy 그래프 그리기
    plot_graphs(train_losses, test_accuracies)


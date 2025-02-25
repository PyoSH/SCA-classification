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

def evaluate_model_device(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)  # GPU로 이동
            labels = labels.to(device)
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

    model.eval()

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

    plot_comparison(y_trues, y_preds, classes=classes)
    # plot_cm(y_trues, y_preds, classes=classes)
    # summary(model, inputs.shape, dtypes=[torch.long])

    return metrics.classification_report(y_trues, y_preds, zero_division=0)

def eval_metrics_device(model, test_loader, classes, device):
    y_trues = None
    y_preds = None
    iter = False

    model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)  # GPU로 이동
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            labels_np = labels.cpu().numpy()
            pred_np = predicted.cpu().numpy()
            if not iter:
                y_trues = labels_np
                y_preds = pred_np
                iter = True
            else:
                y_trues = np.concatenate((y_trues, labels_np), axis=0)
                y_preds = np.concatenate((y_preds, pred_np), axis=0)

    plot_comparison(y_trues, y_preds, classes=classes)
    # plot_cm(y_trues, y_preds, classes=classes)
    # summary(model, inputs.shape, dtypes=[torch.long])

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

def plot_comparison(y_ts, y_ps, classes, show='True'):
    x = np.arange(len(y_ts))

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(x, y_ts, color='red', linestyle='-', marker='', label='GT', linewidth=1)
    plt.xlabel('Audio frame')
    plt.ylabel('Operational situation')
    plt.yticks(ticks=np.arange(len(classes)), labels=classes)
    plt.legend()
    plt.title('Comparison of real-time prediction and GT')

    plt.subplot(2, 1, 2)
    # plt.plot(x, y_ps ,  color='green', linestyle='-', marker='', label='LSTM predicted', linewidth=1)
    plt.plot(x, y_ps, color='blue', linestyle='-', marker='', label='RNN predicted', linewidth=1)
    plt.xlabel('Audio frame')
    plt.ylabel('Operational situation')

    plt.yticks(ticks=np.arange(len(classes)), labels=classes)
    plt.legend()

    plt.tight_layout()
    plt.show()

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    train_losses = []  # 각 epoch의 training loss 기록
    test_accuracies = []  # 각 epoch의 test accuracy 기록

    for epoch in range(num_epochs):
        model.train()  # 모델을 training 모드로 설정

        idx = 0
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 현재 epoch의 평균 training loss 기록
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # 현재 epoch의 test accuracy 계산 및 기록
        test_accuracy = evaluate_model(model, test_loader)
        test_accuracies.append(test_accuracy)
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    # training loss 및 test accuracy 그래프 그리기
    plot_graphs(train_losses, test_accuracies)

def train_model_device(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    train_losses = []  # 각 epoch의 training loss 기록
    test_accuracies = []  # 각 epoch의 test accuracy 기록

    for epoch in range(num_epochs):
        model.train()  # 모델을 training 모드로 설정

        idx = 0
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 현재 epoch의 평균 training loss 기록
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # 현재 epoch의 test accuracy 계산 및 기록
        test_accuracy = evaluate_model_device(model, test_loader, device)
        test_accuracies.append(test_accuracy)
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    # training loss 및 test accuracy 그래프 그리기
    plot_graphs(train_losses, test_accuracies)
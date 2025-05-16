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
import matplotlib.pyplot as plt
from loguru import logger
from datetime import datetime

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

def evaluate_model_device(model, test_loader, criterion, device):
    model.eval()
    correct, total = 0, 0
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    test_loss /= len(test_loader)

    return accuracy, test_loss

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

    plot_comparison(model.name, y_trues, y_preds, classes=classes)
    # plot_cm(model.name, y_trues, y_preds, classes=classes)

    return metrics.classification_report(y_trues, y_preds, zero_division=0)

def eval_metrics_device(model, test_loader, classes, device):
    y_trues = None
    y_preds = None
    iter = False

    model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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

    plot_comparison(model.name, y_trues, y_preds, classes=classes)
    plot_cm(model.name, y_trues, y_preds, classes=classes)

    return metrics.classification_report(y_trues, y_preds, zero_division=0)

def plot_graphs(model_name, train_losses, test_accuracies):
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
    # plt.show()
    time_now = datetime.now().strftime("%H_%M")
    prefix = os.path.join('pics', 'trains')
    file_name_str = f'train_curve_{model_name}_{time_now}.png'
    plt.savefig(os.path.join(prefix, file_name_str))

def plot_cm(model_name, y_true, y_pred, classes, show='True'):

    mapped_y_true = np.array(classes)[y_true]
    mapped_y_pred = np.array(classes)[y_pred]

    cm = confusion_matrix(mapped_y_true, mapped_y_pred, labels=classes)
    print(cm)
    cm_vis = seaborn.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # plt.show()
    prefix = os.path.join('pics', 'trains')
    file_name_str = f'CM_{model_name}.png'
    plt.savefig(os.path.join(prefix, file_name_str))

def plot_comparison(model_type, y_ts, y_ps, classes, show='True'):
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
    if model_type == 'LSTM':
        plt.plot(x, y_ps ,  color='green', linestyle='-', marker='', label='LSTM predicted', linewidth=1)
    elif model_type == 'RNN':
        plt.plot(x, y_ps, color='blue', linestyle='-', marker='', label='RNN predicted', linewidth=1)
    elif model_type == 'CRNN_3':
        plt.plot(x, y_ps, color='cyan', linestyle='-', marker='', label='CRNN predicted', linewidth=1)
    elif model_type == 'CLSTM_3':
        plt.plot(x, y_ps, color='magenta', linestyle='-', marker='', label='CLSTM predicted', linewidth=1)
    elif model_type == 'C_test':
        plt.plot(x, y_ps, color='black', linestyle='-', marker='', label='CLSTM-test predicted', linewidth=1)
    elif 'C-MultiScale' in model_type:
        plt.plot(x, y_ps, color='black', linestyle='-', marker='', label='AMSC-RNN predicted', linewidth=1)

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
    plot_graphs(model.name, train_losses, test_accuracies)

def train_model_device(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    train_losses = []  # 각 epoch의 training loss 기록
    test_losses = []
    test_accuracies = []  # 각 epoch의 test accuracy 기록

    for epoch in range(num_epochs):
        model.train()  # 모델을 training 모드로 설정

        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

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
        test_accuracy, test_loss = evaluate_model_device(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}%')

    # training loss 및 test accuracy 그래프 그리기
    # plot_graphs(model.name, train_losses, test_accuracies)

    return train_losses, test_losses, test_accuracies

def plot_kfold_curves(model_name, all_train_losses, all_test_losses, all_test_accuracies):
    epochs = len(all_train_losses[0])
    x = range(1, epochs + 1)

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    for i, fold_loss in enumerate(all_train_losses):
        fold_loss = np.squeeze(fold_loss)  # (1, N) → (N,)
        plt.plot(x, fold_loss, label=f'Fold {i + 1}')
    plt.title('Train Loss per Fold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    for i, fold_loss in enumerate(all_test_losses):
        plt.plot(x, np.squeeze(fold_loss), label=f'Fold {i + 1}')
    plt.title('Test Loss per Fold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    for i, fold_acc in enumerate(all_test_accuracies):
        plt.plot(x, np.squeeze(fold_acc), label=f'Fold {i+1}')
    plt.title('Test Accuracy per Fold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    prefix = os.path.join('pics', 'trains')
    file_name_str = f'kfold_summary_{model_name}.png'
    plt.savefig(os.path.join(prefix, file_name_str))
    plt.show()

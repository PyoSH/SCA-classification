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
from matplotlib.animation import FuncAnimation

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
    # animate_comparison(model.name, y_trues, y_preds, classes=classes, out_path='B2_small_comp.mp4', fps=10)
    # plot_cm(model.name, y_trues, y_preds, classes=classes)

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
    plt.xlabel('Audio frame', fontsize=14)
    plt.ylabel('Operational situation', fontsize=14)
    plt.yticks(ticks=np.arange(len(classes)), labels=classes)
    plt.tick_params(axis='both', which='major', labelsize=16)
    # plt.legend()
    # plt.title('Comparison of real-time prediction and GT')

    plt.subplot(2, 1, 2)
    if model_type == 'B1':
        plt.plot(x, y_ps, color='#1f77b4', linestyle='-', marker='', label='MFCC+LSTM', linewidth=1)
    elif model_type == 'B2-small':
        plt.plot(x, y_ps, color='#ff7f0e', linestyle='-', marker='', label='SingleScale CNN-small', linewidth=1)
    elif model_type == 'B2-middle':
        plt.plot(x, y_ps, color='#2ca02c', linestyle='-', marker='', label='SingleScale CNN-middle', linewidth=1)
    elif model_type == 'B2-large':
        plt.plot(x, y_ps, color='#d62728', linestyle='-', marker='', label='SingleScale CNN-large', linewidth=1)
    elif model_type == 'B3':
        plt.plot(x, y_ps, color='#9467bd', linestyle='-', marker='', label='MultiScale (Kaiming)', linewidth=1)
    elif model_type == 'B4':
        plt.plot(x, y_ps, color='#8c564b', linestyle='-', marker='', label='MultiScale+Mel init', linewidth=1)
    elif model_type == 'P':
        plt.plot(x, y_ps, color='#e377c2', linestyle='-', marker='', label='Proposed', linewidth=1)
    elif model_type == 'SVM':
        plt.plot(x, y_ps, color='green', linestyle='-', marker='', label='MFCC+SVM', linewidth=1)
    elif model_type == 'AST':
        plt.plot(x, y_ps, color='purple', linestyle='-', marker='', label='AST', linewidth=1)


    # Show the legend
    # plt.legend()

    plt.xlabel('Audio frame', fontsize=14)
    plt.ylabel('Operational situation', fontsize=14)

    plt.yticks(ticks=np.arange(len(classes)), labels=classes)
    plt.tick_params(axis='both', which='major', labelsize=16)
    # plt.legend()

    plt.tight_layout()
    plt.show()
    # plt.savefig("/Users/seunghyunpyo/PycharmProjects/rnn_followup/pics/testings")

from matplotlib.ticker import MaxNLocator

def animate_comparison(model_type, y_ts, y_ps, classes, out_path="comparison.mp4", fps=15):
    x = np.arange(len(y_ts))

    # 색/라벨 매핑
    style_map = {
        'B1':        ("#1f77b4", "B1"),
        'B2-small':  ("#ff7f0e", "B2-small"),
        'B2-middle': ("#2ca02c", "B2-middle"),
        'B2-large':  ("#d62728", "B2-large"),
        'B3':        ("#9467bd", "B3"),
        'B4':        ("#8c564b", "B4"),
        'P':         ("#e377c2", "Proposed"),
    }
    pred_color, pred_label = style_map.get(model_type, ("#1f77b4", model_type))

    # Figure & Axis (하나만)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, len(x) - 1)
    ax.set_ylim(-0.5, len(classes) - 0.5)
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Audio frame')
    ax.set_ylabel('Operational situation')
    ax.set_title(f'Model output: {pred_label}')

    # Prediction 라인
    (pred_line,) = ax.plot([], [], color=pred_color, linewidth=1,
                           label=pred_label, drawstyle='steps-post')

    # 현재 프레임 표시선(선택사항)
    (cursor,) = ax.plot([], [], 'k--', alpha=0.3)

    ax.legend(loc='upper left')

    def init():
        pred_line.set_data([], [])
        cursor.set_data([], [])
        return pred_line, cursor

    def update(frame):
        # 모델 예측 누적 표시
        pred_line.set_data(x[:frame + 1], y_ps[:frame + 1])
        cursor.set_data([frame, frame], [ax.get_ylim()[0], ax.get_ylim()[1]])
        return pred_line, cursor

    ani = FuncAnimation(
        fig, update, frames=len(x), init_func=init,
        blit=True, interval=1000 / fps
    )

    # 저장
    try:
        ani.save(out_path, fps=fps, extra_args=['-vcodec', 'libx264'])
    except Exception:
        from matplotlib.animation import PillowWriter
        gif_path = out_path.rsplit('.', 1)[0] + '.gif'
        ani.save(gif_path, writer=PillowWriter(fps=fps))
        print(f"ffmpeg이 없어 GIF로 저장했습니다: {gif_path}")
    finally:
        plt.close(fig)
    # 사용 예시
    # animate_comparison('P', y_ts, y_ps, classes, out_path='comparison.mp4', fps=15)

def animate_gt(y_ts, classes, out_path="gt.mp4", fps=15):
    x = np.arange(len(y_ts))

    # Figure & Axis
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, len(x)-1)
    ax.set_ylim(-0.5, len(classes)-0.5)
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Audio frame')
    ax.set_ylabel('Operational situation')
    ax.set_title('Ground Truth (GT)')

    # GT 라인 (빨간색)
    (gt_line,) = ax.plot([], [], color='red', linewidth=1,
                         label='GT', drawstyle='steps-post')

    # 현재 프레임 표시선
    (cursor,) = ax.plot([], [], 'k--', alpha=0.3)

    ax.legend(loc='upper left')

    # 초기화
    def init():
        gt_line.set_data([], [])
        cursor.set_data([], [])
        return gt_line, cursor

    # 업데이트
    def update(frame):
        gt_line.set_data(x[:frame+1], y_ts[:frame+1])
        cursor.set_data([frame, frame], [ax.get_ylim()[0], ax.get_ylim()[1]])
        return gt_line, cursor

    ani = FuncAnimation(
        fig, update, frames=len(x), init_func=init,
        blit=True, interval=1000/fps
    )

    # 저장
    try:
        ani.save(out_path, fps=fps, extra_args=['-vcodec', 'libx264'])
    except Exception:
        from matplotlib.animation import PillowWriter
        gif_path = out_path.rsplit('.', 1)[0] + '.gif'
        ani.save(gif_path, writer=PillowWriter(fps=fps))
        print(f"ffmpeg이 없어 GIF로 저장했습니다: {gif_path}")
    finally:
        plt.close(fig)

    # 사용 예시
    # animate_gt(y_ts, classes, out_path='gt.mp4', fps=15)

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
        cnt = 0
        for inputs, labels in train_loader:

            if cnt == len(train_loader)-1:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            cnt +=1

        # 현재 epoch의 평균 training loss 기록
        epoch_loss = running_loss / (len(train_loader)-1)
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

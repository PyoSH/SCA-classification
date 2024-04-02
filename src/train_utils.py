import seaborn
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

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
def evaluate_test(model, test_loader):
    corrects, total, total_loss = 0, 0, 0
    model.eval()
    iter = 0
    for inputs, labels in test_loader:
        logit = model(inputs)
        loss = F.cross_entropy(logit, labels, reduction="sum")
        _, predicted = torch.max(logit.data, 1)
        total += labels.size(0)
        total_loss += loss.item()
        corrects += (predicted == labels).sum()
        iter += 1

    avg_loss = total_loss / len(test_loader.dataset)
    avg_acc = corrects / total
    return avg_loss, avg_acc
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

def plot_cm(y_true, y_pred, annot=True, cmap='Blues', show='True'):
    cm = confusion_matrix(y_true, y_pred)
    seaborn.heatmap(cm, annot, cmap)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if show:
        plt.show()
    else:
        # plt.imsave()
        pass


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

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {test_accuracy:.2f}%')

    # training loss 및 test accuracy 그래프 그리기
    plot_graphs(train_losses, test_accuracies)


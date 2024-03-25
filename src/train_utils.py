import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    train_losses = []  # 각 epoch의 training loss 기록
    test_accuracies = []  # 각 epoch의 test accuracy 기록

    for epoch in range(num_epochs):
        model.train()  # 모델을 training 모드로 설정

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 현재 epoch의 training loss 기록
        train_losses.append(loss.item())

        # 현재 epoch의 test accuracy 계산 및 기록
        test_accuracy = evaluate_model(model, test_loader)
        test_accuracies.append(test_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {test_accuracy:.2f}%')

    # training loss 및 test accuracy 그래프 그리기
    plot_graphs(train_losses, test_accuracies)


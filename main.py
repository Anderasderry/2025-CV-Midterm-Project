import os
import csv
import random
from collections import defaultdict
from tqdm import tqdm
from itertools import product
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from datetime import datetime

# 设置参数
if torch.cuda.is_available():
    print("CUDA is available")
else:
    print("CUDA is not available")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = '101_ObjectCategories'
batch_size = 32
num_classes = 101
train_per_class = 30
seed = 107

# 数据增强与预处理
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform_train)

# 按类别划分训练/测试/验证集
def split_dataset_by_class(dataset, train_per_class=30, val_per_class=6, seed=107):
    random.seed(seed)
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for label, indices in class_indices.items():
        if len(indices) < train_per_class + 1:
            continue
        random.shuffle(indices)
        train_idx.extend(indices[:train_per_class - val_per_class])
        val_idx.extend(indices[train_per_class - val_per_class:train_per_class])
        test_idx.extend(indices[train_per_class:])
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

train_set, val_set, test_set = split_dataset_by_class(dataset, train_per_class=30, val_per_class=6)
train_set.dataset.transform = transform_train
val_set.dataset.transform = transform_eval
test_set.dataset.transform = transform_eval

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 模型构建
def create_model(pretrained=True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

# 单轮训练
def train_one_epoch(model, loader, optimizer, criterion, epoch=None, total_epochs=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    loop = tqdm(loader, desc=f"Epoch [{epoch}/{total_epochs}]" if epoch else "Training", leave=False)
    for x, y in loop:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)

        loop.set_postfix(loss=loss.item(), acc=100. * correct / total)
    return total_loss / total, correct / total

def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

@torch.no_grad()
def evaluate_loss_and_accuracy(model, loader, criterion):
    model.eval()
    correct, total = 0, 0
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

epochs_list = [20, 30]
lr_fc_list = [1e-3, 5e-4]
lr_base_list = [1e-4, 1e-5]

# 组合所有超参数
grid = list(product(epochs_list, lr_fc_list, lr_base_list))

if __name__ == "__main__":
    # 创建 CSV 文件（如果不存在）
    results_file = 'grid_search_results.csv'
    write_header = not os.path.exists(results_file)

    with open(results_file, mode='a', newline='') as f_csv:
        writer_csv = csv.writer(f_csv)
        if write_header:
            writer_csv.writerow(['Experiment', 'Epochs', 'LR_FC', 'LR_Base', 'Best_Val_Acc', 'Test_Acc'])

        # 主训练循环
        for i, (epochs, lr_fc, lr_base) in enumerate(grid):
            print(f"\n实验 {i+1}: epochs={epochs}, lr_fc={lr_fc}, lr_base={lr_base}")
            model = create_model(pretrained=True)

            for param in model.parameters():
                param.requires_grad = True

            optimizer = torch.optim.Adam([
                {'params': model.fc.parameters(), 'lr': lr_fc},
                {'params': [p for name, p in model.named_parameters() if not name.startswith('fc')], 'lr': lr_base}
            ])
            criterion = nn.CrossEntropyLoss()
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

            log_dir = f"runs/grid_exp{i+1}_ep{epochs}_fc{lr_fc}_base{lr_base}_{datetime.now().strftime('%m%d_%H%M%S')}"
            writer = SummaryWriter(log_dir=log_dir)

            best_acc = 0
            for epoch in range(epochs):
                train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, epoch+1, epochs)
                val_loss, val_acc = evaluate_loss_and_accuracy(model, val_loader, criterion)

                print(f"\nEpoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accuracy/val', val_acc, epoch)
                writer.add_scalar('LR/base', optimizer.param_groups[1]['lr'], epoch)
                writer.add_scalar('LR/fc', optimizer.param_groups[0]['lr'], epoch)

                scheduler.step()

                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), f"best_model_exp{i+1}.pth")

            writer.close()

            # 加载最佳模型，评估测试集
            print(f"\n实验 {i+1} 在测试集上的最终表现:")
            model.load_state_dict(torch.load(f"best_model_exp{i+1}.pth"))
            test_loss, test_acc = evaluate_loss_and_accuracy(model, test_loader, criterion)
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

            writer_csv.writerow([i+1, epochs, lr_fc, lr_base, round(best_acc, 4), round(test_acc, 4)])
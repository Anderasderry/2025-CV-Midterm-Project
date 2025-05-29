from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from main import (
    create_model,
    split_dataset_by_class,
    train_one_epoch,
    evaluate_loss_and_accuracy,
    device,
    transform_train,
    transform_eval,
)

# 设置参数
data_dir = '101_ObjectCategories'
batch_size = 32
num_classes = 101
train_per_class = 30
val_per_class = 6
seed = 107

# 加载数据集（注意 transform 设置方式）
dataset_all = datasets.ImageFolder(root=data_dir)
train_set, val_set, test_set = split_dataset_by_class(dataset_all, train_per_class, val_per_class, seed)

# 每个子集绑定 transform（确保互不影响）
train_set.dataset.transform = transform_train
val_set.dataset.transform = transform_eval
test_set.dataset.transform = transform_eval

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 创建模型（不加载预训练权重）
model = create_model(pretrained=False)

# 设置所有参数为可训练
for param in model.parameters():
    param.requires_grad = True

# 超参数（根据最佳组合）
epochs = 30
lr_fc = 1e-3
lr_base = 1e-4

# 优化器和损失函数
optimizer = torch.optim.Adam([
    {'params': model.fc.parameters(), 'lr': lr_fc},
    {'params': [p for name, p in model.named_parameters() if not name.startswith('fc')], 'lr': lr_base}
])
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# TensorBoard 日志记录
log_dir = f"runs/scratch_ep{epochs}_fc{lr_fc}_base{lr_base}_{datetime.now().strftime('%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=log_dir)

# 训练过程
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
        torch.save(model.state_dict(), f"best_model_without_pretrained.pth")

writer.close()

# 测试集评估
print("\n加载 best_model_without_pretrained.pth，在测试集上评估：")
model.load_state_dict(torch.load("best_model_without_pretrained.pth"))
test_loss, test_acc = evaluate_loss_and_accuracy(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

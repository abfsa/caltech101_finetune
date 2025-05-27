import torch
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from dataset import get_caltech101_loaders
import os
from pathlib import Path

def get_model(model_name='resnet18', pretrained=True, num_classes=101):
    model = getattr(models, model_name)(pretrained=pretrained)
    # 替换最后一层
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def train(model, train_loader, test_loader, epochs, lr, device, log_dir):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)  # TensorBoard日志

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        writer.add_scalar('Loss/train', total_loss / len(train_loader), epoch)
        writer.add_scalar('Accuracy/train', acc, epoch)
        print(f'Epoch {epoch+1}: loss={total_loss:.3f}, acc={acc:.3f}')
    
    writer.close()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" 

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_caltech101_loaders(data_dir=DATA_DIR, batch=32)

    # 训练预训练模型
    model_pre = get_model(pretrained=True)
    train(model_pre, train_loader, test_loader, epochs=10, lr=1e-4, device=device, log_dir='../runs/pretrained')

    # 从零开始训练
    model_scratch = get_model(pretrained=False)
    train(model_scratch, train_loader, test_loader, epochs=10, lr=1e-3, device=device, log_dir='caltech101_finetune/runs/scratch')
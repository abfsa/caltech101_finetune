import torch
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from codes.dataset import get_caltech101_loaders
from codes.train import train, get_model
import os
from pathlib import Path

DATA_DIR = 'data'

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_caltech101_loaders(data_dir=DATA_DIR, batch=32)

    # 训练预训练模型
    #model_pre = get_model(pretrained=True)
    #train(model_pre, train_loader, test_loader, epochs=10, lr=1e-4, device=device, log_dir='../runs/pretrained')

    # 从零开始训练
    model_scratch = get_model(pretrained=False)
    train(model_scratch, train_loader, test_loader, epochs=60, lr_bb=1e-2, lr_fc=1e-2, weight_decay=1e-4, 
        device=device, log_dir=Path(r'runs/scratch/lr=1e-2wd=1e-4'), checkpoint_path=Path(rf'models/scratch/lr=1e-2wd=1e-4.pth'))
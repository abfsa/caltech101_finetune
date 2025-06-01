import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, Subset
import random
import torch
from collections import defaultdict

def get_caltech101_loaders(data_dir, batch = 32, seed = 2025): 
    """
    
    """

    torch.manual_seed(seed)
    random.seed(seed)

    #对于caltech101中的部分灰度图，先转成rgb三通道
    rgb_first = transforms.Lambda(lambda img: img.convert("RGB"))
    # 训练集进行数据增强
    transform_train = transforms.Compose([
        rgb_first,
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3,0.3,0.3,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # 测试集不进行数据增强
    transform_test = transforms.Compose([
        rgb_first,
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), 
    ])

    train_full = datasets.Caltech101(root=data_dir, transform=transform_train, download=False, target_type="category")
    test_full = datasets.Caltech101(root=data_dir, transform=transform_test, download=False, target_type="category")

    #没类随机30张作为train，其余为test
    cls_to_indices = defaultdict(list)
    for i, (_, label) in enumerate(train_full):
        cls_to_indices[label].append(i)

    train_indices = []
    test_indices = []
    for indices in cls_to_indices.values():
        random.shuffle(indices)
        train_indices.extend(indices[:30])
        test_indices.extend(indices[30:])
    train_set = Subset(train_full, train_indices)
    test_set  = Subset(test_full, test_indices)
    

    train_loader = DataLoader(train_set, batch_size=batch,
                            shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(test_set,  batch_size=batch*2,
                            shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader
    

if __name__ == "__main__":
    pass
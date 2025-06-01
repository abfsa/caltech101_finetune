import torch
from torch.utils.data import DataLoader
from codes.dataset import Caltech101Dataset
from torchvision.models import resnet50
import argparse
from sklearn.metrics import classification_report

def test(model, test_loader, device):
    model.eval()
    true_labels = []
    pred_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())
    
    print(classification_report(true_labels, pred_labels))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_path", type=str, default="data/caltech101", help="Dataset path")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型（需与训练代码保持一致）
    model = resnet50(num_classes=101)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    
    # 加载测试集
    test_dataset = Caltech101Dataset(args.data_path, split="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    test(model, test_loader, device)

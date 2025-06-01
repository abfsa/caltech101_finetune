import torch
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from codes.dataset import get_caltech101_loaders
from codes.train import train, get_model
import os
from pathlib import Path
import argparse
import yaml
import itertools
import json

DATA_DIR = 'data'

def parse_arguments():
    parser = argparse.ArgumentParser(description='Caltech101 Training')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--data_path', type=str, default= DATA_DIR, help='数据集路径')
    parser.add_argument('--output_dir', type=str, help='模型保存路径')
    # 其他参数保持原有定义...
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    

if __name__ == '__main__':
    args = parse_arguments()
    cfg = load_config(args.config)
    merged_args = {
        **cfg.get('training', {}),
        **vars(args)
    }
    data_path = merged_args.get('data_path') or cfg['data_dir']
    output_dir = merged_args.get('output_dir') or 'models'
    

    tr_loader, val_loader = get_caltech101_loaders(
        data_dir=data_path , batch=cfg["batch"])

    search_space = list(itertools.product(
        cfg["lr_backbone"], cfg["lr_fc"], cfg["weight_decay"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for lr_backbone, lr_fc, wd in search_space:
        tag       = f"bb{lr_backbone:.0e}_fc{lr_fc:.0e}_wd{wd:.0e}"
        log_dir   = Path(merged_args.get('output_dir')) / tag
        ckpt_path = Path(merged_args.get('output_dir')) / f"{tag}.pth"

        print(f"\n=== {tag} ===")
        model = get_model(model_name=cfg["model_name"],
                          pretrained=cfg["pretrained"])

        best_acc = train(model, tr_loader, val_loader,
                         epochs=cfg["epochs"],
                         lr_bb=lr_backbone,
                         lr_fc=lr_fc,
                         weight_decay=wd,
                         device=device,
                         log_dir=log_dir,
                         checkpoint_path=ckpt_path)

        results[tag] = best_acc

    with open(Path(merged_args.get('output_dir')) / "search_results.json", "w") as f:
        json.dump(results, f, indent=2)


import itertools, yaml, time, json
from pathlib import Path
import torch
from codes.train import get_model, train
from codes.dataset import get_caltech101_loaders

BASE_RUN_DIR  = Path("runs/lr_search")
BASE_CKPT_DIR = Path("models/pretrained")
CONFIG_FILE   = "config.yaml"    # 网格定义

def load_cfg(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_cfg(CONFIG_FILE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_loader, val_loader = get_caltech101_loaders(
        data_dir=cfg["data_dir"], batch=cfg["batch"])

    search_space = list(itertools.product(
        cfg["lr_backbone"], cfg["lr_fc"], cfg["weight_decay"]))

    results = {}
    for lr_backbone, lr_fc, wd in search_space:
        tag       = f"bb{lr_backbone:.0e}_fc{lr_fc:.0e}_wd{wd:.0e}"
        log_dir   = BASE_RUN_DIR / tag
        ckpt_path = BASE_CKPT_DIR / f"{tag}.pth"

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

    with open(BASE_RUN_DIR / "search_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

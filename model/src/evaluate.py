"""test 평가: overall ACC, 클래스별 precision/recall/f1, confusion matrix(CSV + PNG), 방향별 ACC.

산출물 (기본: results/<run>/eval_<split>/):
  metrics.json, classification_report.csv, confusion_matrix.csv, confusion_matrix.png, predictions.csv

사용 예:
  python src/evaluate.py --run-name 20261013_230000
  python src/evaluate.py --ckpt checkpoints/xxx/best.pt --split test
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import config as C
from dataset import build_loader
from model import build_model
from train import get_device


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, probs, idxs = [], [], []
    for x, _, i in loader:
        p = torch.softmax(model(x.to(device, non_blocking=True)), 1).cpu()
        probs.append(p)
        preds.append(p.argmax(1))
        idxs.append(i)
    return torch.cat(idxs).numpy(), torch.cat(preds).numpy(), torch.cat(probs).numpy()


def plot_cm(cm, out_path, title):
    labels = C.CLASS_NAMES_EN  # 서버에 한글 폰트가 없을 수 있어 영문 라벨 사용
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-name", help="checkpoints/<run>/best.pt 사용")
    g.add_argument("--ckpt", type=Path)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--split-dir", type=Path, default=C.SPLIT_DIR)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=C.NUM_WORKERS)
    ap.add_argument("--gpu", type=int, default=0, help="-1 이면 CPU")
    args = ap.parse_args()

    ckpt_path = args.ckpt or C.CHECKPOINT_DIR / args.run_name / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    run = cfg.get("run_name", Path(ckpt_path).parent.name)
    out_dir = args.out_dir or C.RESULTS_DIR / run / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.gpu)
    model = build_model(dropout=cfg.get("dropout", C.DROPOUT), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    loader = build_loader(args.split, split_dir=args.split_dir, batch_size=args.batch_size,
                          num_workers=args.num_workers, image_size=cfg.get("image_size", C.IMAGE_SIZE),
                          train=False)
    idx, pred, prob = predict(model, loader, device)
    df = loader.dataset.df.iloc[idx].reset_index(drop=True)
    y = df["label"].to_numpy()
    labels = list(range(C.NUM_CLASSES))

    acc = accuracy_score(y, pred)
    report = classification_report(y, pred, labels=labels, target_names=C.CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)

    df["pred"] = pred
    df["pred_name"] = [C.CLASS_NAMES[p] for p in pred]
    df["correct"] = (pred == y).astype(int)
    for k, name in enumerate(C.CLASS_NAMES_EN):
        df[f"prob_{name}"] = prob[:, k].round(5)
    df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8")

    per_view = {C.VIEW_EN.get(v, v or "unknown"): float(g["correct"].mean())
                for v, g in df.groupby("view", dropna=False)}
    pd.DataFrame(report).T.to_csv(out_dir / "classification_report.csv", encoding="utf-8")
    pd.DataFrame(cm, index=C.CLASS_NAMES, columns=C.CLASS_NAMES).to_csv(
        out_dir / "confusion_matrix.csv", encoding="utf-8")
    plot_cm(cm, out_dir / "confusion_matrix.png", f"{run} [{args.split}] acc={acc:.4f}")

    metrics = {"run": run, "ckpt": str(ckpt_path), "ckpt_epoch": ckpt.get("epoch"), "split": args.split,
               "n": int(len(y)), "accuracy": float(acc), "macro_f1": report["macro avg"]["f1-score"],
               "accuracy_by_view": per_view}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(classification_report(y, pred, labels=labels, target_names=C.CLASS_NAMES, digits=4, zero_division=0))
    print("confusion matrix (행=정답, 열=예측):")
    print(pd.DataFrame(cm, index=C.CLASS_NAMES, columns=C.CLASS_NAMES).to_string())
    print(f"\n[{args.split}] ACC = {acc:.4f}  (n={len(y)})  by view: "
          + ", ".join(f"{k}={v:.4f}" for k, v in per_view.items()))
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()

"""DenseNet121 베이스라인 학습.

산출물:
  results/<run>/train_log.csv   epoch별 train/val loss·acc, lr, 소요시간
  results/<run>/config.json     실행 설정
  checkpoints/<run>/best.pt     best val acc
  checkpoints/<run>/last.pt     마지막 epoch

사용 예:
  python src/train.py --gpu 0
  python src/train.py --gpu 1 --memory-efficient
"""
import argparse
import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config as C
from dataset import build_loader
from model import build_model


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(gpu):
    if gpu < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu}")


def run_epoch(model, loader, criterion, device, optimizer=None, amp=False):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", type=Path, default=C.SPLIT_DIR)
    ap.add_argument("--run-name", default=None, help="기본값: 타임스탬프")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--dropout", type=float, default=C.DROPOUT)
    ap.add_argument("--image-size", type=int, default=C.IMAGE_SIZE)
    ap.add_argument("--num-workers", type=int, default=C.NUM_WORKERS)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--gpu", type=int, default=0, help="-1 이면 CPU")
    ap.add_argument("--amp", action="store_true", help="mixed precision (공식 베이스라인은 미사용)")
    ap.add_argument("--no-pretrained", action="store_true", help="ImageNet 가중치 없이 (오프라인 테스트용)")
    ap.add_argument("--memory-efficient", action="store_true",
                    help="DenseNet gradient checkpointing (공용 GPU 메모리 부족 시)")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed, args.deterministic)
    device = get_device(args.gpu)
    run = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir, ckpt_dir = C.RESULTS_DIR / run, C.CHECKPOINT_DIR / run
    res_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg.update(run_name=run, device=str(device), optimizer=C.OPTIMIZER, class_names=C.CLASS_NAMES,
               torch=torch.__version__)
    (res_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    loader_kw = dict(split_dir=args.split_dir, batch_size=args.batch_size,
                     num_workers=args.num_workers, image_size=args.image_size, seed=args.seed)
    train_loader = build_loader("train", **loader_kw)
    val_loader = build_loader("val", **loader_kw)
    print(f"[info] run={run} device={device} train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    model = build_model(dropout=args.dropout, pretrained=not args.no_pretrained,
                        memory_efficient=args.memory_efficient).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_path = res_dir / "train_log.csv"
    fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time_sec", "is_best"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)

    best_acc, best_epoch = -1.0, -1
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, args.amp)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device, None, args.amp)
        dt = time.time() - t0
        is_best = va_acc > best_acc
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "val_acc": va_acc, "config": cfg}
        if is_best:
            best_acc, best_epoch = va_acc, epoch
            torch.save(state, ckpt_dir / "best.pt")
        torch.save(state, ckpt_dir / "last.pt")

        row = [epoch, f"{tr_loss:.4f}", f"{tr_acc:.4f}", f"{va_loss:.4f}", f"{va_acc:.4f}",
               optimizer.param_groups[0]["lr"], f"{dt:.1f}", int(is_best)]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"[epoch {epoch:02d}/{args.epochs}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {dt:.0f}s{'  *best' if is_best else ''}")

    print(f"[done] best val acc {best_acc:.4f} @ epoch {best_epoch}")
    print(f"  log : {log_path}\n  ckpt: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

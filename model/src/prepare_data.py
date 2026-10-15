"""라벨 JSON 재귀 수집 → (이미지경로, 라벨) 목록 → 클래스 층화 8:1:1 분할 → splits/*.csv

JSON 배치(클래스별 1파일 / 이미지별 1파일)에 무관하게 label_root 아래 *.json 을 전부 읽는다.

사용 예:
  python src/prepare_data.py --data-root /path/to/data
  python src/prepare_data.py --image-root /path/원천데이터 --label-root /path/라벨링데이터
"""
import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import config as C


def nfc(s):
    return unicodedata.normalize("NFC", s) if isinstance(s, str) else s


def load_json(path):
    for enc in ("utf-8-sig", "cp949"):
        try:
            with open(path, encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise ValueError("unsupported encoding")


def iter_annotations(obj):
    """JSON 최상위가 dict/list 어느 쪽이든 annotation dict 들을 꺼낸다."""
    if isinstance(obj, list):
        for o in obj:
            yield from iter_annotations(o)
    elif isinstance(obj, dict):
        anns = obj.get("annotations")
        if isinstance(anns, list):
            for a in anns:
                if isinstance(a, dict):
                    yield a
        elif isinstance(anns, dict):
            yield anns
        elif "diagnosis_info" in obj and "photograph" in obj:
            yield obj  # annotation 자체가 최상위인 경우


def parse_annotation(ann):
    """분류에 필요한 필드만 추출. lesion_area / lesions / bbox 는 사용하지 않으므로 접근하지 않는다."""
    diag = (ann.get("diagnosis_info") or {}).get("diagnosis_name")
    photo = ann.get("photograph") or {}
    rel = photo.get("file_path")
    if not diag or not rel:
        return None
    rel = nfc(str(rel)).replace("\\", "/").lstrip("/")
    parts = rel.split("/")
    view = next((p for p in parts if p in C.VIEWS), "")
    return {
        "identifier": ann.get("identifier") or Path(rel).stem,
        "class_name": nfc(str(diag)).strip(),
        "rel_path": rel,
        "view": view,
        "width": photo.get("width"),
        "height": photo.get("height"),
    }


class ImageResolver:
    """rel_path → 실제 파일 경로. 직접 결합이 실패하면(NFD 한글, 상위 폴더 차이 등) 파일명 인덱스로 찾는다."""

    def __init__(self, image_root):
        self.root = image_root
        self._index = None

    def _build_index(self):
        print(f"[info] 직접 경로 매칭 실패 → {self.root} 아래 이미지 파일명 인덱스 생성 중...")
        idx = {}
        for p in self.root.rglob("*"):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                # 같은 파일명이 여러 곳에 있으면 (마스킹 이미지 등) 경로 정보로 구분
                idx.setdefault(p.name, []).append(p)
        self._index = idx

    def __call__(self, rel):
        for cand in (rel, unicodedata.normalize("NFD", rel)):
            p = self.root / cand
            if p.is_file():
                return p
        if self._index is None:
            self._build_index()
        cands = self._index.get(Path(rel).name, [])
        cands = [c for c in cands if "lesion_area" not in c.parts]
        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1:  # 상대경로 꼬리가 가장 많이 일치하는 후보
            tail = rel.split("/")
            return max(cands, key=lambda c: sum(nfc(a) == b for a, b in zip(reversed(c.parts), reversed(tail))))
        return None


def collect(image_root, label_root):
    json_files = sorted(label_root.rglob("*.json"))
    print(f"[info] label_root={label_root}  → .json {len(json_files)}개 발견")
    if not json_files:
        sys.exit(f"[error] {label_root} 아래에 .json 파일이 없습니다.")

    resolve = ImageResolver(image_root)
    records, bad_json, skipped, unknown_cls, missing_img = [], [], 0, Counter(), []
    for jp in json_files:
        try:
            obj = load_json(jp)
        except Exception as e:  # 깨진/비라벨 JSON 은 건너뜀
            bad_json.append((jp, repr(e)))
            continue
        for ann in iter_annotations(obj):
            r = parse_annotation(ann)
            if r is None:
                skipped += 1
                continue
            if r["class_name"] not in C.CLASS_TO_IDX:
                unknown_cls[r["class_name"]] += 1
                continue
            img = resolve(r["rel_path"])
            if img is None:
                missing_img.append(r["rel_path"])
                continue
            r["label"] = C.CLASS_TO_IDX[r["class_name"]]
            r["image_path"] = str(img.resolve())
            r["json_path"] = str(jp)
            records.append(r)

    df = pd.DataFrame(records)
    n_before = len(df)
    if n_before:
        df = df.drop_duplicates(subset="image_path").reset_index(drop=True)

    print(f"[info] 유효 레코드 {len(df)}개 (중복 제거 {n_before - len(df)}개)")
    if bad_json:
        print(f"[warn] 읽기 실패/비라벨 JSON {len(bad_json)}개 (예: {bad_json[0][0]} {bad_json[0][1]})")
    if skipped:
        print(f"[warn] diagnosis_name 또는 file_path 누락 annotation {skipped}개 건너뜀")
    if unknown_cls:
        print(f"[warn] 알 수 없는 클래스명 건너뜀: {dict(unknown_cls)}")
    if missing_img:
        print(f"[warn] 이미지 파일을 찾지 못함 {len(missing_img)}개 (예: {missing_img[:3]})")
    if df.empty:
        sys.exit("[error] 유효한 레코드가 없습니다.")
    return df


def print_counts(df):
    tab = pd.crosstab(df["class_name"], df["view"].replace("", "?"), margins=True, margins_name="합계")
    tab = tab.reindex([c for c in C.CLASS_NAMES if c in tab.index] + ["합계"])
    print("\n[수집 결과] 클래스 × 방향")
    print(tab.to_string())

    warn = []
    for cname in C.CLASS_NAMES:
        n = int((df["class_name"] == cname).sum())
        if n != C.EXPECTED_PER_CLASS:
            warn.append(f"{cname}={n}")
    if warn:
        print(f"[warn] 클래스당 기대 개수 {C.EXPECTED_PER_CLASS}와 다름: {', '.join(warn)} (계속 진행)")


def split(df, seed):
    tr, va, te = C.SPLIT_RATIO
    # 클래스×방향으로 층화 (클래스 층화를 포함). 그룹이 너무 작으면 클래스만으로 층화
    key = df["label"].astype(str) + "_" + df["view"]
    if key.value_counts().min() < 10:
        key = df["label"]
    train_df, rest = train_test_split(df, test_size=va + te, stratify=key, random_state=seed)
    rest_key = key.loc[rest.index]
    val_df, test_df = train_test_split(rest, test_size=te / (va + te), stratify=rest_key, random_state=seed)
    return train_df, val_df, test_df


def main():
    ap = argparse.ArgumentParser()
    C.add_path_args(ap)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    image_root, label_root = C.resolve_roots(args)
    print(f"[info] image_root={image_root}")

    df = collect(image_root, label_root)
    print_counts(df)

    parts = dict(zip(("train", "val", "test"), split(df, args.seed)))
    args.split_dir.mkdir(parents=True, exist_ok=True)
    cols = ["image_path", "label", "class_name", "view", "identifier", "rel_path", "width", "height", "json_path"]
    for name, d in parts.items():
        d = d.sort_values(["label", "rel_path"])[cols]
        d.to_csv(args.split_dir / f"{name}.csv", index=False, encoding="utf-8")

    dist = pd.DataFrame({n: d["class_name"].value_counts() for n, d in parts.items()}).reindex(
        [c for c in C.CLASS_NAMES if c in df["class_name"].unique()]).fillna(0).astype(int)
    dist.loc["합계"] = dist.sum()
    print(f"\n[분할 결과] seed={args.seed}, ratio={C.SPLIT_RATIO}")
    print(dist.to_string())
    print(f"\n[done] {args.split_dir}/train.csv, val.csv, test.csv 저장")


if __name__ == "__main__":
    main()

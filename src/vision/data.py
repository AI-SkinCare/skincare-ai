"""Read original ZIPs in place; never extract/copy the image collection."""
from __future__ import annotations
import collections
import hashlib
import io
import json
import random
import re
import zipfile
from pathlib import Path

from torch.utils.data import Dataset

CLASSES = ("건선", "아토피", "여드름", "정상", "주사", "지루")
ENGLISH = ("psoriasis", "atopic", "acne", "normal", "rosacea", "seborrheic")
SOURCE_URL = "https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71863"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def read_rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def save_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    tmp.replace(path)


def build_manifest(root, seed=42, dev_fraction=.1):
    root=Path(root).resolve()
    if not 0 < dev_fraction < .5:
        raise ValueError("dev_fraction must be between zero and .5")
    rows=[]; descriptions=collections.defaultdict(collections.Counter); archives={}
    for prefix,origin in (("TL","Training"),("VL","Validation")):
        for path in sorted(root.rglob(prefix+"_*.zip")):
            _,label,view=path.stem.split("_")
            if label not in CLASSES or view not in ("정면","측면"):
                raise ValueError(f"unknown class/view: {path.name}")
            image_prefix="TS" if origin=="Training" else "VS"
            matches=list(root.rglob(image_prefix+"_"+label+"_"+view+".zip"))
            if len(matches)!=1:
                raise ValueError(f"missing/ambiguous source ZIP: {path}")
            image_zip=matches[0]
            with zipfile.ZipFile(image_zip) as images:
                image_members={Path(n.lstrip("/")).name:n for n in images.namelist() if n.lower().endswith(".png")}
                if len(image_members)!=len(images.namelist()):
                    raise ValueError("ambiguous or unexpected image members")
                with zipfile.ZipFile(path) as labels:
                    for member in sorted(labels.namelist()):
                        if not member.endswith(".json"):continue
                        raw=labels.read(member)
                        for a in json.loads(raw.decode("utf-8-sig"))["annotations"]:
                            d=a["diagnosis_info"]
                            if d["diagnosis_name"]!=label:
                                raise ValueError("archive and annotated class disagree")
                            name=Path(a["photograph"]["file_path"]).name
                            info=images.getinfo(image_members[name])
                            identifier=a["identifier"]
                            rows.append({"id":identifier,"origin_split":origin,"label":label,"class_index":CLASSES.index(label),
                                         "view":view,"group":re.sub(r"_P\d+_L\d+$","",identifier),
                                         "image_zip":str(image_zip),"image_member":info.filename,
                                         "image_crc":info.CRC,"image_size_bytes":info.file_size,
                                         "label_zip":str(path),"label_member":member,
                                         "annotation_sha256":hashlib.sha256(raw).hexdigest(),
                                         "medical_description":d.get("desc",a.get("desc",""))})
            for p in (path,image_zip):
                stat=p.stat(); archives[str(p)]={"size":stat.st_size,"mtime_ns":stat.st_mtime_ns}
    if not rows or len({r['id'] for r in rows})!=len(rows):
        raise ValueError("empty dataset or duplicate image IDs")
    # Conservative grouping by inferred source ID, plus identical-byte candidates.
    # ZIP CRC/size is used only for exclusion; it is not a cryptographic identity proof.
    held=[r for r in rows if r["origin_split"]=="Validation"]
    held_groups={r["group"] for r in held}
    held_crc={(r["image_crc"],r["image_size_bytes"]) for r in held}
    excluded=[]; pool=[]
    for r in rows:
        if r["origin_split"]!="Training":continue
        if r["group"] in held_groups or (r["image_crc"],r["image_size_bytes"]) in held_crc:
            excluded.append(r["id"])
        else:pool.append(r)
    # Seeded group partition; the same source group is never split for tuning.
    groups=collections.defaultdict(list)
    for r in pool:groups[r["group"]].append(r)
    ordered=sorted(groups); random.Random(seed).shuffle(ordered)
    dev_groups=set(ordered[:max(1,round(len(ordered)*dev_fraction))])
    # Also exclude training CRC candidates matching internal-dev images.
    dev_crc={(r["image_crc"],r["image_size_bytes"]) for r in pool if r['group'] in dev_groups}
    result=[]
    for r in pool:
        r=dict(r)
        if r['group'] in dev_groups:r['split']='dev'
        elif (r['image_crc'],r['image_size_bytes']) in dev_crc:
            excluded.append(r['id']);continue
        else:r['split']='train'
        result.append(r)
    result += [dict(r,split='holdout') for r in held]
    for split in ('train','dev','holdout'):
        if {r['label'] for r in result if r['split']==split}!=set(CLASSES):
            raise ValueError(f"missing class in {split}")
    report={"rows":len(result),"downloaded_rows":len(rows),"seed":seed,"internal_dev_fraction":dev_fraction,
            "split_counts":dict(collections.Counter(r['split'] for r in result)),
            "class_view_counts":dict(collections.Counter(r['split']+'/'+r['label']+'/'+r['view'] for r in result)),
            "excluded_train_ids":excluded,"source_group_overlap":0,"crc_candidate_overlap_after_exclusion":0,
            "group_identity":"filename prefix before _P..._L...; not verified patient identity",
            "evaluation_scope":"synthetic downloaded Validation used as holdout; no independent clinical test",
            "official_test_downloaded":False,"archives":archives,"dataset_hash":fingerprint(result)}
    # Build definitions after exclusions and the internal dev partition.
    # Validation/dev descriptions must not enter recommendation context.
    for row in result:
        if row['split']=='train' and row['label']!='정상' and row['medical_description']:
            descriptions[ENGLISH[row['class_index']]][row['medical_description']]+=1
    knowledge={"source":SOURCE_URL,"dataset":"AI Hub 71863","split":"train","scope":"class-level definition, not observations about this user",
               "labels":{label:{"general_description":counts.most_common(1)[0][0],"training_records":sum(counts.values()),
                                 "description_variants":len(counts)} for label,counts in descriptions.items()}}
    return result,report,knowledge


class ZipImages(Dataset):
    def __init__(self,rows,size=256):
        from torchvision import transforms
        self.rows=list(rows); self.handles={}
        self.transform=transforms.Compose([transforms.Resize((size,size),antialias=True),transforms.ToTensor(),
                                          transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        from PIL import Image
        r=self.rows[index]
        z=self.handles.get(r['image_zip'])
        if z is None:z=self.handles.setdefault(r['image_zip'],zipfile.ZipFile(r['image_zip']))
        with Image.open(io.BytesIO(z.read(r['image_member']))) as img:
            x=self.transform(img.convert('RGB'))
        return x,r['class_index'],index
    def __getstate__(self):
        return dict(self.__dict__,handles={})
    def close(self):
        for handle in self.handles.values():handle.close()
        self.handles.clear()

"""Train AI Hub-style DenseNet, EfficientNet, and a frozen DINOv2 linear head."""
from __future__ import annotations
import argparse
import collections
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from .data import CLASSES,ZipImages,build_manifest,fingerprint,read_rows,save_json
from .models import FrozenDINO,build_model,checkpoint_config,load_checkpoint


def metrics(labels,predictions,views=None):
    cm=[[0]*len(CLASSES) for _ in CLASSES]
    for y,p in zip(labels,predictions):cm[int(y)][int(p)]+=1
    per={}
    for k,name in enumerate(CLASSES):
        tp=cm[k][k]; support=sum(cm[k]); predicted=sum(row[k] for row in cm)
        per[name]={'support':support,'precision':tp/predicted if predicted else 0.,
                   'recall':tp/support if support else 0.,
                   'f1':2*tp/(support+predicted) if support+predicted else 0.}
    n=len(labels)
    result={'n':n,'accuracy':sum(cm[k][k] for k in range(len(CLASSES)))/n if n else 0.,
            'macro_f1':sum(p['f1'] for p in per.values())/len(CLASSES),'per_class':per,'confusion_matrix':cm}
    if views:
        result['by_view']={v:metrics([y for y,w in zip(labels,views) if w==v],
                                    [p for p,w in zip(predictions,views) if w==v]) for v in sorted(set(views))}
    return result


def loader(dataset,batch,workers=0,shuffle=False,seed=42):
    return DataLoader(dataset,batch_size=batch,shuffle=shuffle,num_workers=workers,
                      pin_memory=torch.cuda.is_available(),persistent_workers=bool(workers),
                      generator=torch.Generator().manual_seed(seed))


def prediction(model,batches,device,cached=False):
    labels=[]; pred=[]; probs=[]; losses=[]
    model.eval()
    with torch.no_grad():
        for x,y,*_ in batches:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):
                logits=model.classifier(x) if cached else model(x)
                loss=nn.functional.cross_entropy(logits.float(),y)
            probabilities=logits.float().softmax(1).cpu()
            losses.append(float(loss)*len(y)); labels.extend(y.cpu().tolist())
            probs.extend(probabilities.tolist());pred.extend(probabilities.argmax(1).tolist())
    return labels,pred,probs,sum(losses)/len(labels)


def feature_cache(model,rows,dataset_hash,path,device,workers,batch):
    if path.exists():
        saved=torch.load(path,weights_only=True,map_location='cpu')
        if saved['dataset_hash']!=dataset_hash or saved['ids']!=[r['id'] for r in rows]:
            raise ValueError('feature cache dataset differs')
        return saved['features']
    features=[]; started=time.perf_counter(); model.eval()
    batches=loader(ZipImages(rows,224),batch,workers)
    with torch.no_grad():
        for i,(x,_,_) in enumerate(batches,1):
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):
                features.append(model.features(x.to(device,non_blocking=True)).float().cpu())
            if i%20==0:print(f'[features] {i}/{len(batches)} {time.perf_counter()-started:.1f}s',flush=True)
    matrix=torch.cat(features)
    torch.save({'dataset_hash':dataset_hash,'ids':[r['id'] for r in rows],'features':matrix},path)
    return matrix


def train_one(arch,rows,report,args):
    folder=args.out/arch; folder.mkdir(parents=True,exist_ok=True)
    run={'dataset_hash':report['dataset_hash'],'architecture':arch,'seed':args.seed,'batch_size':args.batch_size,
         'epochs':args.linear_epochs if arch=='dinov2_linear' else args.cnn_epochs,'lr':.001,
         'optimizer':'Adam','dropout':.5,'selection':'internal dev macro F1; lower loss breaks ties',
         'image_size':224 if arch=='dinov2_linear' else 256,'pretrained':True,
         'torch':str(torch.__version__),'holdout_used_for_selection':False}
    if (folder/'run.json').exists() and json.loads((folder/'run.json').read_text(encoding='utf-8'))!=run:
        raise ValueError('existing run configuration changed; use a new output directory')
    save_json(folder/'run.json',run)
    if (folder/'summary.json').exists():
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        if summary.get('phase')=='complete':
            print(f'[skip] completed {arch}',flush=True);return summary
    torch.manual_seed(args.seed); random.seed(args.seed)
    model=build_model(arch,pretrained=True).to(args.device)
    cached=isinstance(model,FrozenDINO)
    groups={s:[i for i,r in enumerate(rows) if r['split']==s] for s in ('train','dev','holdout')}
    if cached:
        # Cache train/dev only. Holdout is first processed after model selection.
        tv=[r for r in rows if r['split']!='holdout']
        matrix=feature_cache(model,tv,fingerprint(tv),folder/'features.pt',args.device,args.workers,args.batch_size)
        pos={r['id']:i for i,r in enumerate(tv)}
        datasets={s:TensorDataset(matrix[[pos[rows[i]['id']] for i in groups[s]]],
                                 torch.tensor([rows[i]['class_index'] for i in groups[s]])) for s in ('train','dev')}
    else:datasets={s:ZipImages([rows[i] for i in groups[s]],run['image_size']) for s in ('train','dev')}
    train_loader=loader(datasets['train'],256 if cached else args.batch_size,0 if cached else args.workers,True,args.seed)
    dev_loader=loader(datasets['dev'],256 if cached else args.batch_size,0 if cached else args.workers)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
    start=0; best=(-1.,float('-inf')); history=[]
    if (folder/'last.pt').exists():
        state=torch.load(folder/'last.pt',map_location=args.device,weights_only=True)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        start=state['epoch'];best=tuple(state['best']);history=state['history']
        torch.set_rng_state(state['rng'].cpu())
        if torch.cuda.is_available():torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
    began=time.perf_counter()
    for epoch in range(start,run['epochs']):
        # Epoch-local order makes resumed data iteration explicit and reproducible.
        train_loader.generator.manual_seed(args.seed+epoch)
        model.train(); total=0.; count=0; t0=time.perf_counter()
        for x,y,*_ in train_loader:
            x=x.to(args.device,non_blocking=True);y=y.to(args.device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.device.startswith('cuda')):
                logits=model.classifier(x) if cached else model(x)
                loss=nn.functional.cross_entropy(logits.float(),y)
            loss.backward();optimizer.step();total+=float(loss.detach())*len(y);count+=len(y)
        y,p,_,dev_loss=prediction(model,dev_loader,args.device,cached)
        score=metrics(y,p); candidate=(score['macro_f1'],-dev_loss)
        record={'epoch':epoch+1,'train_loss':total/count,'dev_loss':dev_loss,'dev_accuracy':score['accuracy'],
                'dev_macro_f1':score['macro_f1'],'seconds':time.perf_counter()-t0}
        history.append(record)
        state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}
        cfg=checkpoint_config(model,arch,run['image_size'])
        if candidate>best:
            best=candidate
            torch.save({'model':state_dict,'config':cfg,'epoch':epoch+1,'dataset_hash':report['dataset_hash']},folder/'best.pt')
        torch.save({'model':state_dict,'optimizer':optimizer.state_dict(),'epoch':epoch+1,'best':list(best),
                    'history':history,'rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},folder/'last.pt')
        save_json(folder/'progress.json',{'phase':'training','architecture':arch,'completed_epochs':epoch+1,
                                         'total_epochs':run['epochs'],'history':history})
        print(arch,json.dumps(record),flush=True)
    if cached:
        if any(p.requires_grad or p.grad is not None for p in model.backbone.parameters()) or model.backbone.training:
            raise RuntimeError('DINO backbone freeze contract violated')
    del optimizer,model
    if args.device.startswith('cuda'):torch.cuda.empty_cache()
    model,checkpoint=load_checkpoint(folder/'best.pt',args.device)
    hold=[rows[i] for i in groups['holdout']]
    y,p,probs,loss=prediction(model,loader(ZipImages(hold,run['image_size']),args.batch_size,args.workers),args.device)
    score=metrics(y,p,[r['view'] for r in hold])
    summary={'phase':'complete',**run,'selected_epoch':checkpoint['epoch'],'holdout':score,'holdout_loss':loss,
             'history':history,'current_session_seconds':time.perf_counter()-began,
             'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
             'dino_backbone_frozen':True if cached else None,
             'evaluation_scope':report['evaluation_scope']}
    save_json(folder/'summary.json',summary)
    save_json(folder/'holdout_predictions.json',[
        {'id':r['id'],'label':r['label'],'view':r['view'],'predicted':CLASSES[k],'probabilities':v}
        for r,k,v in zip(hold,p,probs)])
    del model
    if args.device.startswith('cuda'):torch.cuda.empty_cache()
    return summary


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('D:/14.안면부 피부질환 이미지 합성데이터'))
    ap.add_argument('--out',type=Path,default=Path('outputs/disease_models_20261107'))
    ap.add_argument('--manifest-out',type=Path,default=Path('data/processed/disease_20261107'))
    ap.add_argument('--architectures',nargs='+',default=['dinov2_linear','efficientnet_b0','densenet121'])
    ap.add_argument('--cnn-epochs',type=int,default=11);ap.add_argument('--linear-epochs',type=int,default=50)
    ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--device',default='cuda');ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args(argv)
    torch.set_num_threads(4)
    if args.cnn_epochs<1 or args.linear_epochs<1:raise ValueError('positive epochs required')
    rows,report,knowledge=build_manifest(args.root,args.seed)
    args.manifest_out.mkdir(parents=True,exist_ok=True);args.out.mkdir(parents=True,exist_ok=True)
    manifest=args.manifest_out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))['dataset_hash']!=report['dataset_hash']:
        raise ValueError('dataset changed; preserve prior outputs and use a new directory')
    save_json(manifest,report);save_json(args.manifest_out/'medical_knowledge.json',knowledge)
    (args.manifest_out/'images.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    print('[dataset]',report['split_counts'],'excluded',len(report['excluded_train_ids']),flush=True)
    if args.prepare_only:return
    summaries=[]
    try:
        for arch in args.architectures:
            save_json(args.out/'status.json',{'phase':'running','architecture':arch,'completed':[s['architecture'] for s in summaries]})
            summaries.append(train_one(arch,rows,report,args))
            save_json(args.out/'comparison.json',summaries)
        save_json(args.out/'status.json',{'phase':'complete','architectures':[s['architecture'] for s in summaries]})
    except Exception as exc:
        save_json(args.out/'failure.json',{'error':type(exc).__name__,'message':str(exc)})
        save_json(args.out/'status.json',{'phase':'failed','error':str(exc)})
        raise


if __name__=='__main__':main()

import copy
import io
import json
import zipfile

import pytest

from server.medical_context import load_medical_knowledge,medical_reference,reference_prompt
from server.schemas import DiseasePrediction
from server.recommendation_quality import sanitize_recommendation


def pred(label='atopic',confidence=.9):
    from server.schemas import DISEASE_LABELS
    probabilities={v:(1-confidence)/5 for v in DISEASE_LABELS.values()}
    probabilities[DISEASE_LABELS[label]]=confidence
    return DiseasePrediction(label=DISEASE_LABELS[label],label_en=label,confidence=confidence,
                             uncertain=False,probabilities=probabilities)


def test_reference_uses_only_predicted_class_and_hides_unobserved_information(tmp_path):
    knowledge={'source':'https://www.aihub.or.kr/example','labels':{
        'atopic':{'general_description':'일반적인 아토피 설명','training_records':100},
        'psoriasis':{'general_description':'일반적인 건선 설명'}}}
    path=tmp_path/'kb.json';path.write_text(json.dumps(knowledge),encoding='utf-8')
    kb=load_medical_knowledge(path)
    ref=medical_reference(pred('psoriasis'),kb,.6)
    assert ref['general_description']=='일반적인 건선 설명'
    assert not ref['severity_assessed'] and not ref['lesion_location_assessed']
    assert not ref['ingredient_efficacy_verified']
    assert 'EASI' in reference_prompt(ref)
    assert medical_reference(pred('atopic',.4),kb,.6) is None
    assert medical_reference(pred('normal'),kb,.6) is None
    assert load_medical_knowledge(tmp_path/'absent')['labels']=={}


@pytest.mark.parametrize('text',['EASI 점수는 18점입니다.','중증도는 중증입니다.','중등도아토피에 해당합니다.','중등도 아토피에 해당합니다.'])
def test_unmeasured_severity_is_removed_from_disease_response(text):
    parsed={'reasoning':[{'content':text}],'answer':text}
    clean,findings=sanitize_recommendation(parsed,disease_context=True)
    assert not clean['answer'] and 'unassessed_severity' in {f['code'] for f in findings}
    assert sanitize_recommendation(parsed,disease_context=False)==(parsed,[])


def test_severity_not_assessed_statement_is_preserved():
    p={'answer':'현재 입력으로 EASI 점수와 중증도를 평가하지 않았습니다.'}
    assert sanitize_recommendation(p,disease_context=True)==(p,[])


def test_recorded_medical_context_response_detox_claim_is_removed():
    text='키틴은 피부 보호막을, 미네랄 솔트는 각질 제거 및 해독에 도움을 줍니다.'
    parsed={'reasoning':[{'content':text}],'answer':text,'ingredients':[{'purpose':text}]}
    clean,findings=sanitize_recommendation(parsed,disease_context=True)
    assert not clean['answer'] and clean['ingredients'][0]['purpose'] is None
    assert {f['code'] for f in findings}=={'unsupported_efficacy'}
    caution={'answer':'미네랄 솔트의 해독에 도움이 된다고 단정할 수 없습니다.'}
    assert sanitize_recommendation(caution)==(caution,[])


def test_recorded_mixed_moisturizer_action_is_not_an_ingredient_purpose():
    from server.postprocess import _purpose
    text='미네랄 솔트가 함유된 보습제를 사용하여 피부 장벽을 강화하고 수분을 공급합니다.'
    assert _purpose(text,'미네랄 솔트') is None
    assert _purpose('미네랄 솔트는 피부 보습에 도움을 줍니다.','미네랄 솔트')=='피부 보습에 도움을 줍니다.'


def test_metrics_match_hand_counted_confusion_matrix_and_views():
    pytest.importorskip('torch')
    from src.vision.train import metrics
    y=[0,0,1,1,2,2,3,3,4,4,5,5]
    p=[0,1,1,1,2,3,3,3,4,5,5,5]
    m=metrics(y,p,['front','side']*6)
    assert m['accuracy']==.75 and m['macro_f1']==pytest.approx((2/3+.8)/2)
    assert m['by_view']['front']['accuracy']==1 and m['by_view']['side']['accuracy']==.5
    assert m['confusion_matrix'][0]==[1,1,0,0,0,0]


def test_zip_manifest_preserves_original_files_and_separates_source_groups(tmp_path):
    pytest.importorskip('torch')
    from src.vision.data import CLASSES,ENGLISH,build_manifest
    for prefix,label_prefix,origin,n in [('TS','TL','Training',100),('VS','VL','Validation',3)]:
        for label,en in zip(CLASSES,ENGLISH):
            ip=tmp_path/origin/(prefix+'_'+label+'_정면.zip')
            lp=tmp_path/origin/(label_prefix+'_'+label+'_정면.zip')
            ip.parent.mkdir(exist_ok=True)
            with zipfile.ZipFile(ip,'w') as images,zipfile.ZipFile(lp,'w') as labels:
                for i in range(n):
                    identifier=f'{en}_{origin}_{i}_P0_L0'
                    images.writestr('/'+identifier+'.png',(identifier+'image').encode())
                    annotation={'annotations':[{'identifier':identifier,'diagnosis_info':{
                        'diagnosis_name':label,'desc':'general '+label},'photograph':{'file_path':identifier+'.png'}}]}
                    labels.writestr('/'+identifier+'.json',json.dumps(annotation))
    rows,report,kb=build_manifest(tmp_path)
    assert report['downloaded_rows']==618
    groups={s:{r['group'] for r in rows if r['split']==s} for s in ('train','dev','holdout')}
    assert not groups['train'] & groups['dev'] and not groups['train'] & groups['holdout']
    assert set(kb['labels'])==set(ENGLISH)-{'normal'}
    assert kb['split']=='train'
    for label,en in zip(CLASSES,ENGLISH):
        if en!='normal':
            assert kb['labels'][en]['training_records']==sum(r['split']=='train' and r['label']==label for r in rows)
    assert all(r['image_member'].startswith('/') for r in rows)
    assert build_manifest(tmp_path)[1]['dataset_hash']==report['dataset_hash']


def test_frozen_dino_head_training_and_grad_cam_do_not_unfreeze_encoder(tmp_path):
    torch=pytest.importorskip('torch')
    pytest.importorskip('torchvision')
    pytest.importorskip('transformers')
    from src.vision.models import build_model,checkpoint_config,load_checkpoint
    from src.vision.explain import grad_cam
    model=build_model('dinov2_linear',dino_config={'hidden_size':24,'num_hidden_layers':2,
                       'num_attention_heads':3,'intermediate_size':48,'image_size':28,'patch_size':14})
    model.train()
    assert not model.backbone.training and all(not p.requires_grad for p in model.backbone.parameters())
    before={k:v.clone() for k,v in model.backbone.state_dict().items()}
    optimizer=torch.optim.Adam(model.classifier.parameters(),lr=.01)
    loss=torch.nn.functional.cross_entropy(model(torch.rand(2,3,28,28)),torch.tensor([0,1]))
    loss.backward();optimizer.step()
    assert all(torch.equal(before[k],v) for k,v in model.backbone.state_dict().items())
    assert all(p.grad is None for p in model.backbone.parameters())
    cam,p,k,layer,available=grad_cam(model,torch.rand(1,3,28,28),'dinov2_linear')
    assert cam.shape==(256,256) and torch.isfinite(cam).all() and 0<=float(cam.min())<=float(cam.max())<=1
    assert abs(sum(p)-1)<1e-5 and 0<=k<6 and layer.endswith('norm1')
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert not model.backbone.encoder.layer[-1].norm1._forward_hooks
    path=tmp_path/'model.pt'
    torch.save({'model':model.state_dict(),'config':checkpoint_config(model,'dinov2_linear',28)},path)
    loaded,_=load_checkpoint(path)
    assert torch.allclose(loaded(torch.ones(1,3,28,28)),model(torch.ones(1,3,28,28)))


@pytest.mark.parametrize('arch',['densenet121','efficientnet_b0'])
def test_cnn_grad_cam_returns_finite_class_specific_map_and_removes_hook(arch):
    torch=pytest.importorskip('torch')
    pytest.importorskip('torchvision')
    from src.vision.models import build_model
    from src.vision.explain import grad_cam,target_layer
    model=build_model(arch).eval()
    cam,p,k,_,_=grad_cam(model,torch.rand(1,3,64,64),arch)
    assert torch.isfinite(cam).all() and cam.shape==(256,256) and len(p)==6
    assert not target_layer(model,arch)[0]._forward_hooks


def test_api_passes_class_reference_to_recommender_and_exposes_explanation(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from server import main
    from server.config import Settings
    from src.llm.inference import MockRecommender
    Image=pytest.importorskip('PIL.Image')
    captured=[]
    class Recording(MockRecommender):
        def recommend(self,x):
            captured.append(x.user_note)
            return super().recommend(x)
    class Classifier:
        version='fixture_model'
        def __init__(self,*args):pass
        def load(self):return self
        def predict(self,_):return pred()
        def explain(self,_):return {'target_class_index':1,'available':True,'method':'grad_cam'}
    monkeypatch.setattr(main,'MockRecommender',Recording)
    monkeypatch.setattr(main,'TrainedDiseaseClassifier',Classifier)
    kb=tmp_path/'kb.json'
    kb.write_text(json.dumps({'source':'https://www.aihub.or.kr/example','labels':{
        'atopic':{'general_description':'일반적인 아토피 설명'}}}),encoding='utf-8')
    cfg=Settings(internal_token='fixture',rec_backend='mock',processed_dir=tmp_path/'absent',
                 disease_backend='dinov2_linear',disease_device='cpu',disease_knowledge_path=kb)
    b=io.BytesIO();Image.new('RGB',(16,16)).save(b,format='PNG')
    with TestClient(main.create_app(cfg)) as client:
        headers={'X-Internal-Token':'fixture'}
        r=client.post('/infer/disease-recommend',headers=headers,json={
            'disease':pred().model_dump(),'survey':{'gender':'female','age':30,'skinType':'normal'}})
        assert r.status_code==200
        assert r.json()['medical_reference']['general_description']=='일반적인 아토피 설명'
        assert '일반적인 아토피 설명' in captured[-1]
        assert 'EASI 점수는 추정하지' in captured[-1]
        r=client.post('/infer/disease-explain',headers=headers,files={'image':('face.png',b.getvalue(),'image/png')})
        assert r.status_code==200 and r.json()['target_label']=='아토피'
        assert client.post('/infer/disease-explain',files={'image':('face.png',b.getvalue(),'image/png')}).status_code==401

"""Reused-photo comparison and real model/Grad-CAM HTTP checks; no training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--photos', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'comparison.json').exists():
        raise RuntimeError('Use a fresh output directory')
    import torch
    from fastapi.testclient import TestClient
    from server.main import create_app
    from server.config import Settings
    from server.disease_classifier import DenseNetDiseaseClassifier, TrainedDiseaseClassifier, CLASS_NAMES
    from src.vision.train import metrics
    torch.set_num_threads(4)
    with (args.photos / 'labels.csv').open(encoding='utf-8-sig', newline='') as handle:
        photos = sorted(csv.DictReader(handle), key=lambda row: row['rel_path'])
    candidates = [('densenet', args.assets / 'model/weights/densenet121_s2.pt')]
    candidates += [(name, args.assets / 'outputs/disease_models_20261107' / name / 'best.pt')
                   for name in ('densenet121', 'efficientnet_b0', 'dinov2_linear')]
    summaries = []
    for architecture, weights in candidates:
        classifier = (DenseNetDiseaseClassifier(weights) if architecture == 'densenet'
                      else TrainedDiseaseClassifier(weights, architecture)).load()
        results = []
        started = time.perf_counter()
        for index, photo in enumerate(photos):
            prediction = classifier.predict((args.photos / photo['rel_path']).read_bytes())
            results.append(dict(case=f'P{index+1:03}', truth=photo['class_name'], view=photo['view'],
                                prediction=prediction.model_dump()))
        y = [CLASS_NAMES.index(row['truth']) for row in results]
        p = [CLASS_NAMES.index(row['prediction']['label']) for row in results]
        summary = dict(architecture=architecture, model_version=classifier.version,
            checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
            metrics=metrics(y, p, [row['view'] for row in results]), seconds=time.perf_counter()-started)
        if architecture == 'dinov2_linear':
            assert not classifier.model.backbone.training
            assert all(not parameter.requires_grad for parameter in classifier.model.backbone.parameters())
            summary['frozen_backbone'] = True
            summary['trainable_parameters'] = sum(p.numel() for p in classifier.model.parameters() if p.requires_grad)
        sample = (args.photos / photos[0]['rel_path']).read_bytes()
        before = {key: tensor.clone() for key, tensor in classifier.model.state_dict().items()}
        cfg = Settings(internal_token='local-mentor-benchmark', rec_backend='mock',
                       disease_backend=architecture, disease_weights=weights, disease_device='cpu',
                       processed_dir=args.assets / 'data/processed/sllm')
        with TestClient(create_app(cfg, disease_classifier_factory=lambda: classifier)) as client:
            response = client.post('/infer/disease', headers={'X-Internal-Token': cfg.internal_token},
                data={'include_explanation': 'true'}, files={'image': ('photo.png', sample, 'image/png')})
            assert response.status_code == 200, response.text
            body = response.json()
            assert body['probabilities'] == results[0]['prediction']['probabilities']
            explanation = body['explanation']
            assert explanation['available'] and explanation['method'] == 'grad_cam'
            assert explanation['model_version'] == body['model_version'] and explanation['target_label'] == body['label']
            assert 'overlay' not in json.dumps(explanation)
            summary['http_explanation'] = dict(available=True, target_label=explanation['target_label'],
                                               map_characters=len(explanation['heatmap_png_base64']))
            assert all(torch.equal(before[key], tensor) for key, tensor in classifier.model.state_dict().items())
            if architecture == 'dinov2_linear':
                assert all(not parameter.requires_grad and parameter.grad is None
                           for parameter in classifier.model.backbone.parameters())
        (args.output / (architecture + '.jsonl')).write_text(
            ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in results), encoding='utf-8')
        summaries.append(summary)
        print(architecture, summary['metrics']['accuracy'], summary['metrics']['by_view']['정면']['accuracy'], flush=True)
        del classifier, before
    (args.output / 'comparison.json').write_text(json.dumps(dict(
        scope='96 reused photos, not a new patient validation or clinical diagnosis score',
        photo_count=len(photos), candidates=summaries), ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

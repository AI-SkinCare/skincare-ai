import base64
import io

import pytest
from fastapi.testclient import TestClient
Image = pytest.importorskip('PIL.Image')

from server.config import Settings
from server.main import create_app
from server.schemas import DiseasePrediction, DiseaseExplanation, DISEASE_LABELS


def png(mode='L'):
    buffer = io.BytesIO()
    Image.new(mode, (256, 256)).save(buffer, format='PNG')
    return buffer.getvalue()


class Classifier:
    version = 'synthetic-classifier-test'

    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def predict(self, _):
        self.calls.append('predict')
        return DiseasePrediction(label='아토피', label_en='atopic', confidence=.8, uncertain=False,
            probabilities={name: .8 if name == '아토피' else .04 for name in DISEASE_LABELS.values()},
            model_version=self.version)

    def explain(self, _):
        self.calls.append('explain')
        if self.failure == 'exception':
            raise RuntimeError('private image or path must never be returned')
        return dict(target_class_index=0 if self.failure == 'target' else 1,
            available=self.failure != 'constant', method='grad_cam',
            probabilities=[.04, .8, .04, .04, .04, .04],
            heatmap_png_base64=base64.b64encode(png('RGB' if self.failure == 'color' else 'L')).decode(),
            overlay_png_base64='must-never-leave-the-server')


def client(classifier):
    cfg = Settings(internal_token='mentor-test', rec_backend='mock', disease_backend='test-factory')
    return TestClient(create_app(cfg, disease_classifier_factory=lambda: classifier))


def test_opt_in_explanation_matches_the_same_classification_and_excludes_overlay():
    classifier = Classifier()
    with client(classifier) as http:
        common = dict(headers={'X-Internal-Token': 'mentor-test'}, files={'image': ('test.png', png(), 'image/png')})
        plain = http.post('/infer/disease', **common).json()
        assert classifier.calls == ['predict'] and plain['explanation'] is None
        body = http.post('/infer/disease', data={'include_explanation': 'true'}, **common).json()
        assert classifier.calls == ['predict', 'predict', 'explain']
        explanation = body.pop('explanation')
        plain.pop('explanation')
        body.pop('inference_ms'); plain.pop('inference_ms')
        assert body == plain and explanation['available']
        assert explanation['target_label'] == body['label']
        assert explanation['model_version'] == body['model_version']
        assert 'overlay' not in str(explanation) and 'must-never' not in str(explanation)
        assert http.post('/infer/disease-explain', **common).json()['available']
        assert http.post('/infer/disease-explain', files=common['files']).status_code == 401


@pytest.mark.parametrize('failure,reason', [('exception', 'failed'), ('color', 'failed'),
                                           ('target', 'failed'), ('constant', 'constant_map')])
def test_optional_explanation_failure_does_not_fail_or_change_classification(failure, reason):
    with client(Classifier(failure)) as http:
        response = http.post('/infer/disease', headers={'X-Internal-Token': 'mentor-test'},
            data={'include_explanation': 'true'}, files={'image': ('test.png', png(), 'image/png')})
        body = response.json()
        assert response.status_code == 200 and body['label'] == '아토피'
        assert body['explanation']['unavailable_reason'] == reason
        assert body['explanation']['heatmap_png_base64'] is None
        assert 'private image' not in response.text


def test_map_schema_rejects_face_overlay_color_and_large_data():
    base = dict(available=True, target_label='아토피', model_version='fixture')
    for data in (base64.b64encode(png('RGB')).decode(), 'x' * 120_001):
        with pytest.raises(ValueError):
            DiseaseExplanation(**base, heatmap_png_base64=data)

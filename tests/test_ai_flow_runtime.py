"""Exercise real queue lifetimes and early API exits without a GPU."""
import asyncio
import json
import threading

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from server import main
from server.config import Settings
from server.ingredient_evidence import load_evidence
from server.main import AppError, Gate
from src.llm.inference import MockRecommender


async def until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(.001)
    await asyncio.wait_for(poll(), 2)


def test_queue_timeout_never_starts_expired_work():
    async def run():
        gate = Gate(1, 1); started = threading.Event(); release = threading.Event(); calls = []
        def hold():
            started.set(); release.wait(2)
        first = asyncio.create_task(gate.run(hold, timeout=3))
        try:
            await until(started.is_set)
            with pytest.raises(AppError) as error:
                await asyncio.wait_for(gate.run(lambda: calls.append('expired'), timeout=.04), .5)
            assert error.value.code == 'E3002' and not calls and gate.waiting == 0
            assert gate.sem.locked()
        finally:
            release.set(); await first
        assert await gate.run(lambda: 'next', timeout=1) == 'next'
    asyncio.run(run())


def test_full_queue_and_cancelled_waiter_do_not_leak_slots():
    async def run():
        gate = Gate(1, 1); started = threading.Event(); release = threading.Event()
        def hold():
            started.set(); release.wait(2)
        first = asyncio.create_task(gate.run(hold, timeout=3))
        try:
            await until(started.is_set)
            queued = asyncio.create_task(gate.run(lambda: 'queued', timeout=1))
            await until(lambda: gate.waiting == 1)
            with pytest.raises(AppError) as error:
                await gate.run(lambda: 'overflow', timeout=1)
            assert error.value.code == 'E3003'
            queued.cancel()
            with pytest.raises(asyncio.CancelledError): await queued
            assert gate.waiting == 0 and gate.sem.locked()
        finally:
            release.set(); await first
        assert await gate.run(lambda: 'next', timeout=1) == 'next'
    asyncio.run(run())


@pytest.mark.parametrize('cancel', [False, True])
def test_running_worker_retains_slot_and_late_error_is_consumed(cancel):
    async def run():
        gate = Gate(1, 0); started = threading.Event(); release = threading.Event(); errors = []
        loop = asyncio.get_running_loop(); previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _, context: errors.append(context))
        def fail_later():
            started.set(); release.wait(2); raise RuntimeError('late worker error')
        task = asyncio.create_task(gate.run(fail_later, timeout=1 if cancel else .04))
        try:
            await until(started.is_set)
            if cancel: task.cancel()
            with pytest.raises(asyncio.CancelledError if cancel else AppError): await task
            assert gate.sem.locked()
            with pytest.raises(AppError) as error: await gate.run(lambda: 'overlap', timeout=1)
            assert error.value.code == 'E3003'
            release.set(); await until(lambda: not gate.sem.locked())
            assert await gate.run(lambda: 'next', timeout=1) == 'next'
            assert not errors
        finally:
            release.set(); loop.set_exception_handler(previous)
    asyncio.run(run())


@pytest.mark.parametrize('kwargs', [dict(rec_concurrency=0), dict(skin_concurrency=0),
    dict(queue_limit=-1), dict(rec_timeout_s=0), dict(rec_timeout_s=float('nan')),
    dict(rec_timeout_s=float('inf')), dict(max_image_mb=0)])
def test_invalid_runtime_settings_fail_at_start(kwargs):
    with pytest.raises(ValueError): Settings(**kwargs)


@pytest.mark.parametrize('endpoint', ['/infer/recommend', '/infer/analyze'])
def test_empty_evidence_skips_model_on_general_routes(monkeypatch, endpoint):
    calls = []
    class Recorder(MockRecommender):
        def recommend(self, x):
            calls.append(x); return super().recommend(x)
    monkeypatch.setattr(main, 'MockRecommender', Recorder)
    survey = dict(gender='female', age=33, skinType='dry', selfReportedConcerns=['건조함'],
                  avoidIngredients=[c['ingredient'] for c in load_evidence()['claims']])
    cfg = Settings(internal_token='test', rec_backend='mock', rec_evidence_mode='enforce', disease_backend='disabled')
    with TestClient(main.create_app(cfg)) as client:
        if endpoint.endswith('/recommend'):
            r = client.post(endpoint, headers={'X-Internal-Token':'test'}, json={'survey':survey})
            rec = r.json()
            assert rec['generation_origin'] == 'not_generated' and rec['inference_ms'] == 0
            assert rec['quality_status'] == rec['evidence_status'] == 'withheld'
            assert not rec['ingredients'] and not rec['routine']
        else:
            r = client.post(endpoint, headers={'X-Internal-Token':'test'}, data={'survey':json.dumps(survey)},
                            files={'image':('x.jpg', b'\xff\xd8\xff' + b'0'*16, 'image/jpeg')})
            assert r.json()['recommendation_status'] == 'quality_rejected'
            assert r.json()['recommendation'] is None
        assert r.status_code == 200 and not calls


@pytest.mark.parametrize('endpoint', ['/infer/analyze', '/infer/disease-analyze'])
def test_form_validation_does_not_reflect_survey(endpoint):
    cfg = Settings(internal_token='test', rec_backend='mock', disease_backend='disabled')
    with TestClient(main.create_app(cfg)) as client:
        r = client.post(endpoint, headers={'X-Internal-Token':'test'},
                        data={'survey':json.dumps({'gender':'PRIVATE_INVALID', 'note':'PRIVATE_NOTE'})},
                        files={'image':('x.jpg', b'bad', 'image/jpeg')})
        assert r.status_code == 422 and r.json()['error']['code'] == 'E4002'
        assert 'PRIVATE_' not in r.text
        assert all('input' not in d and 'ctx' not in d for d in r.json()['error']['detail'])


def test_oversized_upload_is_read_only_to_limit(monkeypatch):
    sizes = []; original = UploadFile.read
    async def read(self, size=-1):
        sizes.append(size); return await original(self, size)
    monkeypatch.setattr(UploadFile, 'read', read)
    cfg = Settings(internal_token='test', rec_backend='mock', max_image_mb=.001)
    with TestClient(main.create_app(cfg)) as client:
        r = client.post('/infer/disease', headers={'X-Internal-Token':'test'},
                        files={'image':('x.jpg', b'\xff\xd8\xff' + b'0'*4096, 'image/jpeg')})
        assert r.status_code == 413 and r.json()['error']['code'] == 'E2003'
        assert sizes == [int(cfg.max_image_mb*1024*1024)+1]

import collections
import copy
import json

import pytest

from src.llm.clean_targets import TargetCleaner
from src.llm.ingredients import IngredientExtractor, harvest
from src.llm.output_filter import OutputFilter
from src.llm.prepare_quality_data import build_dataset, normalize_steps, serving_key, validate_splits
from src.llm.schema import RecInput, build_target


def test_distinct_plants_parts_and_numeric_names_are_not_merged():
    ext = IngredientExtractor({"CYMBOPOGONSCHOENANTHUSLEAF/STEMEXTRACT": "대나무추출물",
                               "LITCHICHINENSISFRUITEXTRACT": "리치 씨앗 가루", "ROOTEXTRACT": "뿌리추출물",
                               "대나무수": "대나무추출물"})
    assert ext.extract("CYMBOPOGON SCHOENANTHUS LEAF/STEM EXTRACT, BAMBUSA VULGARIS LEAF/STEM EXTRACT") == ["카멜그라스잎/줄기추출물", "대나무잎/줄기추출물"]
    assert ext.extract("LITCHI CHINENSIS FRUIT EXTRACT, LITCHI CHINENSIS SEED POWDER") == ["리치열매추출물", "리치 씨앗 가루"]
    assert ext.extract("ROOT EXTRACT, SEED EXTRACT") == []
    assert ext.extract("대나무수 BAMBUSA VULGARIS WATER") == ["대나무수"]
    assert ext.extract("헥사펩타이드-11, 3-o-에틸아스코빅애씨드, 1-메틸하이단토인-2-이미드") == ["헥사펩타이드-11", "3-o-에틸아스코빅애씨드", "1-메틸하이단토인-2-이미드"]
    assert ext.extract("헥사펩타이드-111") == []
    assert ext.extract("황사, 상황, 황은 피지를") == ["유황"]


def test_shared_korean_alias_does_not_bridge_different_english_identities():
    lex, _ = harvest(["공유오일(ALPHA OIL), 첫오일(ALPHA OIL)",
                      "공유오일(BETA OIL), 두번째오일(BETA OIL)"])
    assert "공유오일" not in lex
    ext = IngredientExtractor(lex)
    assert ext.extract("ALPHA OIL BETA OIL") == ["첫오일", "두번째오일"]


def test_cleaner_preserves_standalone_and_alias_names_and_flags_conflicts():
    cleaner = TargetCleaner(OutputFilter.from_yaml(), {})
    out = cleaner.clean_text("또한, (SULFUR)는 피지를 조절합니다. 황(SULFUR)은 좋아요.", collections.Counter())
    assert "(유황)은" in out and "유황은" in out and "또한,는" not in out
    bad = "대나무잎/줄기추출물(CYMBOPOGON SCHOENANTHUS LEAF/STEM EXTRACT)은 좋아요."
    assert cleaner.conflicting_pairs(bad)
    assert cleaner.korean_names(bad, collections.Counter()) == bad


def test_spacing_in_representative_name_does_not_create_loss_or_false_conflict():
    old = {"COCHLEARIAARMORACIAROOTEXTRACT": "서양고추냉이 뿌리 추출물",
           "서양고추냉이뿌리추출물": "서양고추냉이 뿌리 추출물"}
    ext = IngredientExtractor(old)
    cleaner = TargetCleaner(OutputFilter.from_yaml(), old)
    text = "서양 고추냉이 뿌리 추출물(COCHLEARIA ARMORACIA ROOT EXTRACT)을 사용하세요."
    assert not cleaner.conflicting_pairs(text)
    cleaned = cleaner.clean_text(text, collections.Counter())
    assert ext.extract(text) == ext.extract(cleaned) == ["서양고추냉이뿌리추출물"]


def row(identifier, age, name="아줄렌", concern="모공", split="train"):
    x = RecInput(gender="여성", age=age, skin_type="중성", concerns=[concern])
    target = build_target([{"content": f"{age}세 사용자입니다."},
                           {"content": f"'{name}'을 권합니다."}, {"content": "사용 전 확인하세요."}], f"{name}을 사용하세요.")
    return {"id": identifier, "split": split, "target_concern": concern, "input": x.to_dict(),
            "target": target, "gold_ingredients": [name]}


def test_builder_quarantines_does_not_leak_and_is_reproducible():
    a, b, bad = row("a", 30), row("b", 40), row("bad", 50)
    bad["target_concern"] = "주름"
    dupe = copy.deepcopy(a)
    dupe["id"] = "a#dup0"
    val = [row("dev", 60, split="val"), row("test", 70, split="val"), row("overlap", 30, split="val")]
    first = build_dataset([a, b, bad, dupe], val, {"dev"})
    second = build_dataset([b, a, dupe, bad], val, {"dev"})
    assert first == second
    assert first["report"]["original_train"] == 3
    assert first["report"]["quarantine_reasons"]["target_concern_not_in_input"] == 1
    assert [r["id"] for r in first["splits"]["test"]] == ["test"]
    assert [r["id"] for r in first["splits"]["val_eval"]] == ["dev"]
    validate_splits(first["splits"])
    for rows in first["splits"].values():
        for r in rows:
            assert r["input"]["user_note"] is None and r["input"]["condition_codes"] is None
            assert len(json.loads(r["target"])["reasoning"]) == 3


def test_validation_only_names_do_not_change_lexicon_or_training_frequencies():
    train = [row("a", 30)]
    unseen = row("v", 40, "새로운오일(ZETA OIL)", split="val")
    a = build_dataset(train, [], set())
    b = build_dataset(train, [unseen], set())
    assert a["lexicon"] == b["lexicon"] and a["freq"] == b["freq"] and a["top"] == b["top"]
    assert b["report"]["quarantine_reasons"]["empty_gold"] == 1


def test_reordered_features_and_hidden_fields_still_count_as_input_overlap():
    a = row("a", 30)
    b = copy.deepcopy(a)
    a["input"]["concerns"] = ["모공", "주름"]
    b["input"]["concerns"] = ["주름", "모공"]
    b["input"]["user_note"] = "추가 정보"
    b["input"]["condition_codes"] = "arbitrary"
    assert serving_key(a) == serving_key(b)
    with pytest.raises(ValueError, match="split leakage"):
        validate_splits({"train": [a], "test": [b]})


def test_fourth_reasoning_step_is_preserved_in_third():
    t = json.loads(row("a", 30)["target"])
    t["reasoning"].append({"content": "추가 주의사항을 보존합니다."})
    normalized = normalize_steps(t)
    assert len(normalized["reasoning"]) == 3
    assert "추가 주의사항" in normalized["reasoning"][2]["content"]

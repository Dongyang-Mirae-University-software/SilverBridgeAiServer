from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.routers import medication_ocr_router as router_module
from app.schemas.medication_ocr_schema import MedicationOcrResponse, OcrDrug
from app.services.drug_name_list import DEFAULT_PATH, DrugNameList, normalize

from tests.test_medication_ocr import FakeService, _make_app, _post


@pytest.fixture(scope="module")
def names() -> DrugNameList:
    loaded = DrugNameList.load(DEFAULT_PATH)
    assert loaded.available and len(loaded) > 30000  # 저장소에 커밋된 실제 목록
    return loaded


# 2026-10-22 실제 봉투 3장 OCR 결과에서 고른 기준 사례 - 실제 목록 파일로 고정한다.
@pytest.mark.parametrize(
    "read_name",
    ["클로리드정250밀리그램", "파탁정20mg", "나소프라정500/20m", "신일이부프로펜정400m", "코푸시럽", "펜클리어맥스정"],
)
def test_in_list(names, read_name):
    assert names.check(read_name) == (True, [])


@pytest.mark.parametrize(
    ("read_name", "expected_prefix"),
    [("펜쿨리어맥스정", "펜클리어맥스정"), ("파박정20mg", "파탁정"), ("펌젠에페리손서방정", "팜젠에페리손서방정")],
)
def test_misread_suggests_correct_name(names, read_name, expected_prefix):
    in_list, suggestions = names.check(read_name)
    assert in_list is False
    assert 1 <= len(suggestions) <= 3
    assert any(s.startswith(expected_prefix) for s in suggestions)
    assert all("(" not in s and "[" not in s for s in suggestions)
    assert len(set(suggestions)) == len(suggestions)


def test_unknown_name_false(names):
    in_list, suggestions = names.check("코쏘열정")
    assert in_list is False
    assert len(suggestions) <= 3


def test_missing_file_gives_null(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        missing = DrugNameList.load(tmp_path / "none.txt")
    assert not missing.available
    assert missing.check("파탁정20mg") == (None, [])
    assert sum("허가 목록" in r.getMessage() for r in caplog.records) == 1


def test_empty_file_gives_null(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("\n\n", encoding="utf-8")
    assert DrugNameList.load(path).check("파탁정") == (None, [])


def test_normalize_rules():
    assert normalize("신일이부프로펜정400밀리그람(수출명:Shinil Ibuprofen)") == "신일이부프로펜정"
    assert normalize("신일이부프로펜정400m") == "신일이부프로펜정"
    assert normalize("나소프라정500/20밀리그램") == "나소프라정"
    assert normalize("A-B_C/D·E,F.G 정") == "abcdefg정"
    assert normalize("[수출용]타이레놀(아세트아미노펜(미분화))정\n") == "타이레놀정"
    assert normalize("비타민0.5mcg정") == "비타민정"


def test_short_prefix_is_not_a_match():
    names = DrugNameList({normalize("정로환"): "정로환", normalize("타이레놀정"): "타이레놀정"})
    assert names.check("정")[0] is False  # 한 글자는 앞부분 일치로 보지 않는다
    assert names.check("타이레")[0] is True
    assert names.check("정로환") == (True, [])  # 정확 일치는 길이 무관


def test_dose_only_name_is_false_not_null():
    names = DrugNameList({normalize("타이레놀정"): "타이레놀정"})
    assert names.check("20mg") == (False, [])


def test_annotate_keeps_original_name():
    names = DrugNameList({normalize("펜클리어맥스정"): "펜클리어맥스정"})
    result = MedicationOcrResponse(
        readable=True, drugs=[OcrDrug(name="펜쿨리어맥스정", confidence="low"), OcrDrug(name="펜클리어맥스정", confidence="high")]
    )
    annotated = names.annotate(result)
    assert [d.name for d in annotated.drugs] == ["펜쿨리어맥스정", "펜클리어맥스정"]
    assert annotated.drugs[0].inList is False and annotated.drugs[0].suggestions == ["펜클리어맥스정"]
    assert annotated.drugs[1].inList is True and annotated.drugs[1].suggestions == []


def _result(*drug_names: str) -> MedicationOcrResponse:
    return MedicationOcrResponse(readable=True, drugs=[OcrDrug(name=n, confidence="high") for n in drug_names])


def test_endpoint_returns_fields_and_logs_counts_only(caplog):
    names = DrugNameList({normalize("펜클리어맥스정"): "펜클리어맥스정"})
    service = FakeService(result=_result("펜쿨리어맥스정", "펜클리어맥스정"))
    with caplog.at_level(logging.INFO, logger=router_module.__name__):
        with TestClient(_make_app(service, drug_names=names)) as client:
            res = _post(client)
    assert res.status_code == 200
    drugs = res.json()["drugs"]
    assert drugs[0]["inList"] is False and drugs[0]["suggestions"] == ["펜클리어맥스정"]
    assert drugs[1]["inList"] is True and drugs[1]["suggestions"] == []
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "drugs=2 not_in_list=1" in text
    assert "펜" not in text


def test_endpoint_without_list_returns_null(tmp_path):
    service = FakeService(result=_result("파탁정20mg"))
    with TestClient(_make_app(service, drug_names=DrugNameList.load(Path(tmp_path) / "none.txt"))) as client:
        res = _post(client)
    assert res.status_code == 200
    assert res.json()["drugs"][0]["inList"] is None
    assert res.json()["drugs"][0]["suggestions"] == []

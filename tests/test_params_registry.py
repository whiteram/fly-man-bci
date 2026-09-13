"""The parameter registry is the single source of truth: the pipeline CAL
and the background defaults must be assembled from it, every CAL key must
have a documented value, and the generated PARAMS.md must be up to date."""

from pathlib import Path

from ffbm import params, pipeline, vizprep


def test_cal_assembled_from_registry():
    cal = params.cal()
    assert cal == pipeline.CAL
    assert cal["I_MID_BASE"] == 90.0 and cal["I_T4_BASE"] == 175.0
    assert cal["OU"]["T45"] == 60.0 and cal["LIF"]["L"] == (20.0, 2.0)


def test_bg_defaults_from_registry():
    assert vizprep.BG_DEFAULTS == params.bg_defaults()
    assert vizprep.BG_DEFAULTS["alpha_hz"] == 10.0


def test_every_section_has_title_and_every_entry_has_4_fields():
    for key, sec in params.SECTIONS.items():
        assert "_title" in sec, f"section {key} missing _title"
        for k, v in sec.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    assert len(v2) == 2, f"{k}.{k2} dict rows: (value, unit)"
            else:
                assert len(v) == 4, f"{k} needs (value, unit, status, note)"
                assert v[2] in ("dataset", "literature", "calibrated",
                                "phenomenol.", "assumed", "numerical",
                                "chosen", "deferred"), f"{k} bad status"


def test_generated_markdown_is_current():
    doc = Path(params.__file__).resolve().parents[2] / "docs" / "PARAMS.md"
    assert doc.exists(), "run `python -m ffbm.params` to generate it"
    assert doc.read_text(encoding="utf-8") == params.to_markdown(), \
        "docs/PARAMS.md is stale -- run `python -m ffbm.params`"

"""Pure-Python unit tests for the escalation feature-engineering logic.

No Spark, no MLflow, no Databricks runtime required — these tests import the
production logic straight from `notebooks/escalation_features.py` and exercise
it on small hand-built pandas DataFrames.

Run from the repo root:

    pytest escalation/tests/test_features.py -v

or from inside `escalation/`:

    pytest tests/test_features.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make `escalation/notebooks/escalation_features.py` importable regardless of
# the working directory pytest is invoked from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_NOTEBOOKS = os.path.abspath(os.path.join(_HERE, "..", "notebooks"))
if _NOTEBOOKS not in sys.path:
    sys.path.insert(0, _NOTEBOOKS)

import escalation_features as ef  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: build a silver-shaped calls DataFrame for one ticket.
# ---------------------------------------------------------------------------
_BASE_TS = pd.Timestamp("2025-09-01 00:00:00")

_ALL_COLS = [
    ef.ID_COL,
    ef.DATETIME_COL,
    "total_call_duration",
    "total_nonspeech_duration",
    "agent_talk_duration",
    "customer_talk_duration",
    "weighted_average",
    "empathy",
    "agent_sentiment",
    "customer_sentiment",
    ef.TARGET_COL,
    "repeat_call_flag",
    "scapia_category",
    "scapia_sub_category",
    "qrc_category",
    "qrc_sub_category",
]


def make_calls(rows, ticket_id="T1"):
    """rows: list of dicts of partial call fields. Missing fields get defaults."""
    records = []
    for i, r in enumerate(rows):
        rec = {
            ef.ID_COL: ticket_id,
            ef.DATETIME_COL: r.get("dt", _BASE_TS + pd.Timedelta(hours=i)),
            "total_call_duration": r.get("total_call_duration", 100),
            "total_nonspeech_duration": r.get("total_nonspeech_duration", 10),
            "agent_talk_duration": r.get("agent_talk_duration", 40),
            "customer_talk_duration": r.get("customer_talk_duration", 30),
            "weighted_average": r.get("weighted_average", 7.0),
            "empathy": r.get("empathy", "neutral"),
            "agent_sentiment": r.get("agent_sentiment", "neutral"),
            "customer_sentiment": r.get("customer_sentiment", "neutral"),
            ef.TARGET_COL: r.get("real_time_alert", "no"),
            "repeat_call_flag": r.get("repeat_call_flag", False),
            "scapia_category": r.get("scapia_category", "flights"),
            "scapia_sub_category": r.get("scapia_sub_category", "refund"),
            "qrc_category": r.get("qrc_category", "complaint"),
            "qrc_sub_category": r.get("qrc_sub_category", "Flight Related"),
        }
        records.append(rec)
    return pd.DataFrame(records, columns=_ALL_COLS)


# ---------------------------------------------------------------------------
# 1. worst-of aggregation
# ---------------------------------------------------------------------------
def test_worst_of_aggregation():
    # bad must win over neutral/good regardless of order or nulls.
    assert ef.worst_of(["good", "neutral", "bad"]) == "bad"
    assert ef.worst_of(["bad", "good"]) == "bad"
    assert ef.worst_of(["neutral", "good"]) == "neutral"
    assert ef.worst_of(["good", "positive"]) in ("good", "positive")  # same rank
    # nulls / sentinels are ignored.
    assert ef.worst_of(["-", None, "good"]) == "good"
    # nothing rankable -> None.
    assert ef.worst_of(["-", None, "inconclusive"]) is None

    # end-to-end through the record builder: one 'bad' customer_sentiment call
    # among neutrals => gl_customer_sentiment == 'bad'.
    calls = make_calls(
        [
            {"customer_sentiment": "neutral"},
            {"customer_sentiment": "bad"},
            {"customer_sentiment": "good"},
        ]
    )
    rec = ef.build_ticket_feature_record(calls)
    assert rec["gl_customer_sentiment"] == "bad"


# ---------------------------------------------------------------------------
# 2. pre-escalation slice: calls after the first 'yes' are excluded
# ---------------------------------------------------------------------------
def test_pre_escalation_slice():
    calls = make_calls(
        [
            {"real_time_alert": "no", "total_call_duration": 100},
            {"real_time_alert": "no", "total_call_duration": 200},
            {"real_time_alert": "yes", "total_call_duration": 999},  # escalation
            {"real_time_alert": "no", "total_call_duration": 999},   # after -> excluded
        ]
    )
    rec = ef.build_ticket_feature_record(calls)
    assert rec is not None
    assert rec[ef.LABEL_COL] == 1
    # Only the 2 pre-escalation calls are considered.
    assert rec["num_calls_considered"] == 2
    # The 999-duration escalation + trailing call must not leak into the mean.
    assert rec["total_call_duration"] == pytest.approx(150.0)
    # Metadata still reflects the full ticket.
    assert rec["total_calls_in_ticket"] == 4


# ---------------------------------------------------------------------------
# 3. tickets whose FIRST call already escalated are dropped
# ---------------------------------------------------------------------------
def test_first_call_escalation_excluded():
    calls = make_calls(
        [
            {"real_time_alert": "yes"},  # first call is already escalated
            {"real_time_alert": "no"},
        ]
    )
    assert ef.build_ticket_feature_record(calls) is None

    # A single-call ticket that IS the escalation is likewise dropped.
    calls_single = make_calls([{"real_time_alert": "yes"}])
    assert ef.build_ticket_feature_record(calls_single) is None


# ---------------------------------------------------------------------------
# 4. neg_sentiment_ratio: proportion of 'bad' customer-sentiment calls
# ---------------------------------------------------------------------------
def test_neg_sentiment_ratio():
    # direct helper: 2 of 4 bad -> 0.5
    assert ef.neg_sentiment_ratio(["bad", "bad", "good", "neutral"], 4) == pytest.approx(0.5)
    # none bad -> 0.0
    assert ef.neg_sentiment_ratio(["good", "neutral"], 2) == pytest.approx(0.0)
    # sentinel handling: '-' is not 'bad'
    assert ef.neg_sentiment_ratio(["bad", "-", None], 3) == pytest.approx(1.0 / 3.0)

    # end-to-end: 1 bad of 3 pre-escalation calls -> 1/3
    calls = make_calls(
        [
            {"customer_sentiment": "bad", "real_time_alert": "no"},
            {"customer_sentiment": "good", "real_time_alert": "no"},
            {"customer_sentiment": "neutral", "real_time_alert": "no"},
        ]
    )
    rec = ef.build_ticket_feature_record(calls)
    assert rec["neg_sentiment_ratio"] == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# 5. speech ratios: mean over calls of talk/total (mean-of-ratios)
# ---------------------------------------------------------------------------
def test_speech_ratios():
    calls = make_calls(
        [
            {
                "total_call_duration": 100,
                "agent_talk_duration": 40,
                "customer_talk_duration": 30,
            },
            {
                "total_call_duration": 200,
                "agent_talk_duration": 40,
                "customer_talk_duration": 100,
            },
        ]
    )
    rec = ef.build_ticket_feature_record(calls)
    # agent: mean(40/100, 40/200) = mean(0.4, 0.2) = 0.3
    assert rec["agent_speech_ratio"] == pytest.approx(0.3)
    # customer: mean(30/100, 100/200) = mean(0.3, 0.5) = 0.4
    assert rec["customer_speech_ratio"] == pytest.approx(0.4)


def test_speech_ratio_zero_duration_call_skipped():
    # A call with total_call_duration == 0 is skipped, not a divide-by-zero.
    calls = make_calls(
        [
            {"total_call_duration": 0, "agent_talk_duration": 10},
            {"total_call_duration": 100, "agent_talk_duration": 50},
        ]
    )
    rec = ef.build_ticket_feature_record(calls)
    assert rec["agent_speech_ratio"] == pytest.approx(0.5)  # only the valid call


# ---------------------------------------------------------------------------
# 6. customer_sentiment_trend: rank(last) - rank(first)
# ---------------------------------------------------------------------------
def test_customer_sentiment_trend():
    # bad(0) -> good(2) : improving = +2
    assert ef.customer_sentiment_trend(["bad", "neutral", "good"]) == pytest.approx(2.0)
    # good(2) -> bad(0) : worsening = -2
    assert ef.customer_sentiment_trend(["good", "bad"]) == pytest.approx(-2.0)
    # fewer than 2 rankable -> 0.0
    assert ef.customer_sentiment_trend(["good"]) == pytest.approx(0.0)
    assert ef.customer_sentiment_trend(["-", None]) == pytest.approx(0.0)
    # positive treated as rank 2 (same as good) -> neutral(1) to positive(2) = +1
    assert ef.customer_sentiment_trend(["neutral", "positive"]) == pytest.approx(1.0)

    # end-to-end preserves call order by datetime.
    calls = make_calls(
        [
            {"customer_sentiment": "bad", "dt": _BASE_TS + pd.Timedelta(hours=2)},
            {"customer_sentiment": "good", "dt": _BASE_TS + pd.Timedelta(hours=1)},
        ]
    )
    # sorted by dt: good(first) -> bad(last) = -2
    rec = ef.build_ticket_feature_record(calls)
    assert rec["customer_sentiment_trend"] == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# 7. calls_per_day with zero time span (no divide-by-zero)
# ---------------------------------------------------------------------------
def test_calls_per_day_zero_duration():
    # Two calls at the SAME timestamp -> time_diff_hours == 0.
    ts = pd.Timestamp("2025-09-01 05:00:00")
    calls = make_calls(
        [
            {"dt": ts, "real_time_alert": "no"},
            {"dt": ts, "real_time_alert": "no"},
        ]
    )
    rec = ef.build_ticket_feature_record(calls)
    assert rec["time_diff_hours"] == pytest.approx(0.0)
    # max(0/24, 1/24) = 1/24 day -> 2 calls / (1/24) = 48, finite (no inf/NaN).
    assert np.isfinite(rec["calls_per_day"])
    assert rec["calls_per_day"] == pytest.approx(48.0)


# ---------------------------------------------------------------------------
# 8. QRC casing normalization: 'Flight Related' == 'flight related'
# ---------------------------------------------------------------------------
def test_qrc_casing_normalization():
    assert ef.normalize_qrc_subcategory("Flight Related") == "Flight Related"
    assert ef.normalize_qrc_subcategory("flight related") == "Flight Related"
    assert ef.normalize_qrc_subcategory("FLIGHT RELATED") == "Flight Related"
    # both variants collapse to a single normalized key
    assert ef.normalize_qrc_subcategory("Flight Related") == ef.normalize_qrc_subcategory(
        "flight related"
    )
    # sentinels normalize to None
    assert ef.normalize_qrc_subcategory("-") is None
    assert ef.normalize_qrc_subcategory(None) is None


# ---------------------------------------------------------------------------
# Bonus coverage: repeat-call parse + worst-of, latest-non-null, target encode,
# and F2 threshold selection — all frozen-encoder logic the pyfunc relies on.
# ---------------------------------------------------------------------------
def test_parse_repeat_call_flag():
    assert ef.parse_repeat_call_flag("Repeat Call: Yes\nReason: ...") is True
    assert ef.parse_repeat_call_flag("Repeat Call: No\nReason: N/A") is False
    assert ef.parse_repeat_call_flag("garbage") is None
    assert ef.parse_repeat_call_flag(None) is None


def test_gl_repeat_calls_worst_of():
    # A repeat call (True) is the 'bad' signal and must win worst-of.
    calls = make_calls(
        [
            {"repeat_call_flag": False},
            {"repeat_call_flag": True},
            {"repeat_call_flag": False},
        ]
    )
    rec = ef.build_ticket_feature_record(calls)
    assert rec["gl_repeat_calls"] == "bad"


def test_latest_non_null_preserves_case():
    # latest non-null keeps the ORIGINAL display casing; sentinels are skipped.
    assert ef.latest_non_null(["A", "-", "B", None]) == "B"
    assert ef.latest_non_null(["-", None]) is None


def test_nonspeech_duration_clamped():
    # Negative non-speech duration is clamped to 0 before averaging.
    calls = make_calls(
        [
            {"total_nonspeech_duration": -50},
            {"total_nonspeech_duration": 100},
        ]
    )
    rec = ef.build_ticket_feature_record(calls)
    # mean(max(-50,0)=0, 100) = 50
    assert rec["total_nonspeech_duration"] == pytest.approx(50.0)


def test_unlabelled_ticket_dropped_in_training():
    # All-null target -> dropped in training, kept at inference.
    calls = make_calls(
        [
            {"real_time_alert": "-"},
            {"real_time_alert": "inconclusive"},
        ]
    )
    assert ef.build_ticket_feature_record(calls, training=True) is None
    rec_inf = ef.build_ticket_feature_record(calls, training=False)
    assert rec_inf is not None
    assert rec_inf[ef.LABEL_COL] == 0  # never escalated


def test_kfold_target_encode_unseen_maps_to_global_mean():
    rng = np.random.default_rng(0)
    n = 200
    cats = rng.choice(["a", "b", "c"], size=n)
    X = pd.DataFrame({"qrc_category": cats})
    # label correlates with category 'c'
    y = pd.Series((cats == "c").astype(int) ^ (rng.random(n) < 0.1).astype(int))
    X_te = pd.DataFrame({"qrc_category": ["a", "b", "c", "UNSEEN_CATEGORY"]})
    _, X_te_enc, encode_maps, global_mean = ef.kfold_target_encode(
        X, y, X_te, ["qrc_category"], n_splits=5, smoothing=10.0
    )
    # Unseen category falls back to the global mean.
    assert X_te_enc["qrc_category"].iloc[-1] == pytest.approx(global_mean)
    assert "qrc_category" in encode_maps
    assert 0.0 <= global_mean <= 1.0


def test_best_f2_threshold_prefers_recall():
    # Perfectly separable at 0.5; ensure a valid threshold on the grid is picked.
    y_true = [0, 0, 1, 1, 1]
    y_proba = [0.1, 0.2, 0.6, 0.7, 0.8]
    thr, f2 = ef.best_f2_threshold(y_true, y_proba)
    assert 0.1 <= thr <= 0.9
    assert f2 == pytest.approx(1.0)  # a threshold in (0.2, 0.6] gives perfect F2


def test_build_model_matrix_frozen_order_and_encoding():
    # A minimal gold row -> model matrix with frozen columns + encodings.
    medians = {c: 0.0 for c in ef.NUMERIC_FEATURES}
    encode_maps = {c: {"x": 0.4} for c in ef.CATEGORICAL_HIGH_CARD_FEATURES}
    gold = pd.DataFrame(
        [
            {
                **{c: 1.0 for c in ef.NUMERIC_FEATURES},
                "gl_empathy": "bad",
                "gl_agent_sentiment": "good",
                "gl_customer_sentiment": "neutral",
                "gl_repeat_calls": "bad",
                "scapia_category": "x",
                "scapia_sub_category": "UNSEEN",
                "qrc_category": "x",
                "qrc_sub_category": "x",
            }
        ]
    )
    X = ef.build_model_matrix(gold, medians, encode_maps, global_mean=0.2)
    # frozen column order
    assert list(X.columns) == ef.FEATURE_NAMES
    # GL ordinal encodings
    assert X["gl_empathy"].iloc[0] == 0.0       # bad
    assert X["gl_agent_sentiment"].iloc[0] == 2.0  # good
    assert X["gl_customer_sentiment"].iloc[0] == 1.0  # neutral
    # high-card: seen -> 0.4, unseen -> global_mean 0.2
    assert X["scapia_category"].iloc[0] == pytest.approx(0.4)
    assert X["scapia_sub_category"].iloc[0] == pytest.approx(0.2)

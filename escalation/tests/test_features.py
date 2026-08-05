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
    ef.CALL_ID_COL,
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
            # Default call_id is unique+ordered per row so it never perturbs the primary datetime sort; tests
            # that exercise tie-breaking pass explicit call_id values.
            ef.CALL_ID_COL: r.get("call_id", f"{ticket_id}-call-{i:04d}"),
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


# ===========================================================================
# REGRESSION TESTS for the cross-review blockers (item 12).
# These would have FAILED against the pre-fix code (target leakage / nondeterminism).
# ===========================================================================


# --- 12(a): OOF target encoding EXCLUDES each row's own label (item 1) ------
def test_oof_target_encode_excludes_own_label():
    """The TRAINING encoding must be out-of-fold: a row's own label must NOT enter its own encoded value.

    Construct a category present in exactly ONE fold-worth of rows with an extreme label, and assert the OOF
    encoding of those rows differs from the full-map encoding (which DOES include their own labels). If the
    training path used the full maps (the pre-fix bug), these would be equal.
    """
    rng = np.random.default_rng(7)
    n = 250
    cats = rng.choice(["a", "b", "c"], size=n)
    X = pd.DataFrame({"qrc_category": cats})
    # 'c' perfectly predicts the label -> full-map encoding for 'c' rows is ~1.0 (includes their own labels),
    # but OOF folds that hold out some 'c' rows pull their encoding toward the global mean.
    y = pd.Series((cats == "c").astype(int))

    X_oof = ef.kfold_oof_encode(X, y, ["qrc_category"], n_splits=5, smoothing=10.0, random_state=42)
    full_maps, gm = ef.compute_target_encode_maps(X, y, ["qrc_category"], smoothing=10.0)
    X_full = ef.apply_target_encode(X["qrc_category"], full_maps["qrc_category"], gm)

    # The two encodings must DIFFER on the training rows (proves OOF is not just the full map).
    assert not np.allclose(X_oof["qrc_category"].to_numpy(), X_full.to_numpy()), (
        "OOF training encoding equals the full-map encoding — that is target leakage (item 1)."
    )
    # And OOF must be deterministic across repeated runs (fixed seed).
    X_oof2 = ef.kfold_oof_encode(X, y, ["qrc_category"], n_splits=5, smoothing=10.0, random_state=42)
    assert np.allclose(X_oof["qrc_category"].to_numpy(), X_oof2["qrc_category"].to_numpy())

    # The FULL/serving map is still the right thing for a HELD-OUT fold (labels not in the map): a fresh test
    # frame encodes to the full-map values, and unseen categories fall back to the global mean.
    X_te = pd.DataFrame({"qrc_category": ["a", "b", "c", "ZZZ"]})
    te_enc = ef.apply_target_encode(X_te["qrc_category"], full_maps["qrc_category"], gm)
    assert te_enc.iloc[-1] == pytest.approx(gm)  # unseen -> global mean


def test_oof_single_fold_holdout_pulls_toward_global_mean():
    """A sharper OOF check: for a category whose label is constant (all 1), the full map encodes it near 1.0,
    while each OOF fold — trained on the OTHER folds — still sees that category as all-1 and also encodes ~1;
    so instead we verify the MECHANISM: an OOF value for a row never equals the smoothed mean that INCLUDES
    that row when removing it changes the fold statistics. We use a category with mixed labels split across
    folds so leave-fold-out changes the estimate."""
    # 10 rows of category 'm': 5 ones, 5 zeros, deterministically interleaved.
    cats = ["m"] * 10 + ["other"] * 40
    labels = ([1, 0] * 5) + list(np.resize([1, 0], 40))
    X = pd.DataFrame({"qrc_category": cats})
    y = pd.Series(labels)
    X_oof = ef.kfold_oof_encode(X, y, ["qrc_category"], n_splits=5, smoothing=0.0, random_state=1)
    full_maps, gm = ef.compute_target_encode_maps(X, y, ["qrc_category"], smoothing=0.0)
    # With zero smoothing the full-map value for 'm' is its overall mean (0.5). At least one OOF-encoded 'm'
    # row must deviate from 0.5, proving the row's own fold was excluded.
    m_oof = X_oof.loc[X["qrc_category"] == "m", "qrc_category"].to_numpy()
    assert not np.allclose(m_oof, full_maps["qrc_category"]["m"]), "OOF did not exclude own-fold labels."


def test_oof_prior_excludes_held_out_fold_label():
    """BLOCKING-1 regression: a held-out row's OOF encoding must be INDEPENDENT of its own label — including
    via the smoothing prior. Flip one held-out row's label (X unchanged, so KFold membership is identical);
    that row's encoding must not change.

    Uses a RARE category so the smoothing prior dominates its encoding — exactly where a dataset-wide prior
    (the old bug) leaked the held-out label. This test FAILS on the old `global_mean = y.mean()` prior (the
    flipped row's encoding shifts by the leaked prior term) and PASSES once the prior is per-fold.
    """
    # 90 'common' (all label 0) + 10 'rare' (mixed) — smoothing=10 makes the prior dominate 'rare' rows.
    X = pd.DataFrame({"qrc_category": ["common"] * 90 + ["rare"] * 10})
    y = pd.Series([0] * 90 + [1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    target = 90  # a 'rare', held-out row

    enc_before = ef.kfold_oof_encode(
        X, y, ["qrc_category"], n_splits=5, smoothing=10.0, random_state=42
    )["qrc_category"]

    y_flipped = y.copy()
    y_flipped.iloc[target] = 1 - y_flipped.iloc[target]
    enc_after = ef.kfold_oof_encode(
        X, y_flipped, ["qrc_category"], n_splits=5, smoothing=10.0, random_state=42
    )["qrc_category"]

    # The flipped row's OWN encoding must be identical (its label — via stats AND prior — is out-of-fold).
    assert enc_before.iloc[target] == pytest.approx(enc_after.iloc[target], abs=1e-12), (
        "Held-out row's OOF encoding changed when its own label flipped — the smoothing prior is leaking the "
        "held-out fold's labels (BLOCKING-1)."
    )


# --- 12(b): deterministic pre-escalation slicing under TIED timestamps (item 4) ---
def test_pre_escalation_slice_deterministic_under_tied_timestamps():
    """Same calls, shuffled input order, tied timestamps -> identical slice + features (call_id tie-break)."""
    ts = pd.Timestamp("2025-09-01 05:00:00")
    # Four calls all at the SAME timestamp; the 3rd (by call_id order) escalates.
    rows = [
        {"call_id": "c1", "dt": ts, "customer_sentiment": "good", "real_time_alert": "no"},
        {"call_id": "c2", "dt": ts, "customer_sentiment": "neutral", "real_time_alert": "no"},
        {"call_id": "c3", "dt": ts, "customer_sentiment": "bad", "real_time_alert": "yes"},
        {"call_id": "c4", "dt": ts, "customer_sentiment": "bad", "real_time_alert": "no"},
    ]
    calls = make_calls(rows)

    rec_ordered = ef.build_ticket_feature_record(calls)
    # Shuffle the input rows — the deterministic (datetime, call_id) sort must recover the same slice.
    shuffled = calls.sample(frac=1.0, random_state=123).reset_index(drop=True)
    rec_shuffled = ef.build_ticket_feature_record(shuffled)

    assert rec_ordered is not None and rec_shuffled is not None
    # Pre-escalation slice = c1, c2 (strictly before c3). c4 (after) is excluded.
    assert rec_ordered["num_calls_considered"] == 2
    assert rec_ordered["is_escalated"] == 1
    # Every feature must be identical regardless of input order.
    for k in ef.FEATURE_NAMES:
        assert rec_ordered[k] == rec_shuffled[k] or (
            pd.isna(rec_ordered[k]) and pd.isna(rec_shuffled[k])
        ), f"feature {k} differs under reordering: {rec_ordered[k]} vs {rec_shuffled[k]}"
    # gl_customer_sentiment over {good, neutral} pre-slice = neutral (bad c3/c4 are excluded).
    assert rec_ordered["gl_customer_sentiment"] == "neutral"


def test_missing_call_id_fails_closed():
    """BLOCKING-2: if the per-call id column is absent, the feature builder must FAIL CLOSED (raise), not
    silently fall back to nondeterministic timestamp-only ordering."""
    calls = make_calls([{"real_time_alert": "no"}, {"real_time_alert": "no"}])
    calls_no_id = calls.drop(columns=[ef.CALL_ID_COL])
    with pytest.raises(ValueError):
        ef.build_ticket_feature_record(calls_no_id)


# The columns the OLD (partial) tie-break hashed — kept here only to PROVE the new hash is not a subset hash.
_OLD_SUBSET_COLS = [
    "Ticket Id",
    "DateTime",
    "Transcript",
    "Real Time Alert (Answer)",
    "Weighted Average (Normalised Score)",
    "Customer Sentiment",
]


def test_content_hash_is_full_row_total_order():
    """BLOCKING-2 residual: the dedup tie-break (`row_content_hash`) must be a TRUE TOTAL ORDER over the FULL
    row. Two rows IDENTICAL on the old 6-field subset but DIFFERING in an un-hashed column (here
    `Agent Talk Duration`) must get DIFFERENT hashes — a subset hash would tie them and let dedup fall back to
    nondeterministic Spark row order.

    This test FAILS if the implementation reverts to hashing only a column subset, and PASSES for the
    full-row hash.
    """
    # Two rows agreeing on every field the old subset hashed, differing ONLY in an un-hashed column.
    base = {
        "Ticket Id": "T1",
        "DateTime": "01-09-2025 05:00:00",
        "Transcript": "hello",
        "Real Time Alert (Answer)": "No",
        "Weighted Average (Normalised Score)": "7.0",
        "Customer Sentiment": "Neutral",
        "Agent Talk Duration (In Seconds)": "40",
        "Empathy (Answer)": "Good",
        "QRC (Category)": "Complaint",
    }
    row_a = dict(base)
    row_b = dict(base, **{"Agent Talk Duration (In Seconds)": "999"})  # differs in an un-hashed column

    # Full-row hash distinguishes them...
    h_a = ef.row_content_hash(row_a)
    h_b = ef.row_content_hash(row_b)
    assert h_a != h_b, "full-row content hash must differ when ANY column differs (tie-break not total order)"

    # ...but the OLD subset hash would NOT (this is exactly the residual bug we fixed).
    h_a_subset = ef.row_content_hash(row_a, columns=_OLD_SUBSET_COLS)
    h_b_subset = ef.row_content_hash(row_b, columns=_OLD_SUBSET_COLS)
    assert h_a_subset == h_b_subset, "sanity: the old 6-field subset genuinely ties these rows"

    # Byte-identical rows collapse (same hash) — genuine duplicates.
    assert ef.row_content_hash(dict(base)) == ef.row_content_hash(dict(base))

    # Deterministic total order: sorting rows by (call_id, content_hash) is stable regardless of input order.
    rows = [row_b, row_a]  # deliberately "wrong" order
    order1 = sorted(rows, key=lambda r: ef.row_content_hash(r))
    order2 = sorted(list(reversed(rows)), key=lambda r: ef.row_content_hash(r))
    assert [ef.row_content_hash(r) for r in order1] == [ef.row_content_hash(r) for r in order2]

    # Null-safety: None != "" != "NULL" (distinct sentinels -> distinct hashes).
    assert ef.row_content_hash({"x": None}) != ef.row_content_hash({"x": ""})
    assert ef.row_content_hash({"x": None}) != ef.row_content_hash({"x": "NULL"})


# The exact magic-string the PRE-FIX content hash used to represent NULL before serialization. Hardcoded here
# (not imported) precisely because the fix REMOVED that constant — a real field value equal to this literal
# used to collide with a genuine NULL.
_OLD_NULL_SENTINEL = " __NULL__ "


def test_content_hash_null_not_collidable_with_sentinel_string():
    """B2 tripwire: a genuine NULL must NOT hash the same as a field whose ACTUAL value equals the old
    magic-string NULL sentinel (" __NULL__ ").

    The old impl serialized NULL as the literal " __NULL__ " and a present value as ``str(value)``, so a real
    call whose field literally held " __NULL__ " produced IDENTICAL bytes to a NULL in that field and could be
    mis-collapsed as a duplicate. The (is_null, value) type-tag makes that impossible: NULL -> "1:", present
    value -> "0:"+value. This assertion FAILS on the old magic-string impl (equal hashes) and PASSES on the
    type-tag fix.
    """
    # Two rows differing ONLY in that field: one a real NULL, the other the literal old-sentinel string.
    row_null = {"ticket_id": "T1", "note": None}
    row_literal = {"ticket_id": "T1", "note": _OLD_NULL_SENTINEL}
    assert ef.row_content_hash(row_null) != ef.row_content_hash(row_literal), (
        "a genuine NULL collides with a field literally equal to the old magic-string sentinel — dedup could "
        "mis-collapse a real call as a duplicate (B2)."
    )

    # Single-field form makes the collision class unambiguous.
    assert ef.row_content_hash({"x": None}) != ef.row_content_hash({"x": _OLD_NULL_SENTINEL})

    # And it must still hold for the OTHER near-miss null-ish literals (defence in depth).
    for literal in (_OLD_NULL_SENTINEL, "__NULL__", "1:", "0:", "None", "null"):
        assert ef.row_content_hash({"x": None}) != ef.row_content_hash({"x": literal}), (
            f"NULL collided with the literal string {literal!r}"
        )

    # A field that genuinely holds the sentinel string is still equal to ITSELF (it is a real, hashable value).
    assert ef.row_content_hash({"x": _OLD_NULL_SENTINEL}) == ef.row_content_hash({"x": _OLD_NULL_SENTINEL})


# --- H4 residual: champion gate is STRUCTURAL + fail-closed (no free-text parsing) ---
class _FakeRegisteredModel:
    def __init__(self, aliases):
        self.aliases = aliases  # list of objects with .alias/.version, OR a dict, OR None/unexpected


class _FakeModelNoAliasesAttr:
    """A model object that has NO `aliases` attribute at all (an unavailable/unexpected shape)."""


class _FakeAliasObj:
    def __init__(self, alias, version):
        self.alias = alias
        self.version = version


class _FakeClient:
    """Minimal MLflow-client stand-in for resolve_alias_version. `get_registered_model` returns a model with
    a configurable alias set (or raises)."""

    def __init__(self, model=None, model_error=None):
        self._model = model
        self._model_error = model_error

    def get_registered_model(self, name):
        if self._model_error is not None:
            raise self._model_error
        return self._model


def test_champion_gate_absent_alias_returns_none():
    """Alias provably absent from a RECOGNIZED, well-formed alias SET -> None ('no champion'), without any
    free-text parsing."""
    client = _FakeClient(model=_FakeRegisteredModel(aliases=[_FakeAliasObj("challenger", "3")]))
    assert ef.resolve_alias_version(client, "cat.sch.model", "champion") is None
    # A well-formed set that has OTHER aliases but not champion is still 'no champion' (dict shape).
    client_dict = _FakeClient(model=_FakeRegisteredModel(aliases={"challenger": "4"}))
    assert ef.resolve_alias_version(client_dict, "cat.sch.model", "champion") is None


def test_champion_gate_wellformed_empty_alias_set_returns_none_first_registration():
    """FIRST-REGISTRATION path (must NOT be blocked): a model that EXISTS but has a well-formed EMPTY alias set
    (champion genuinely absent) reads as 'no champion' -> None, WITHOUT raising. This is the legitimate case
    the fail-closed logic must preserve, distinct from the unknown/None-shape fail-closed cases below.
    """
    # Empty list AND empty dict are both well-formed empties -> None, no raise.
    client_list = _FakeClient(model=_FakeRegisteredModel(aliases=[]))
    assert ef.resolve_alias_version(client_list, "cat.sch.model", "champion") is None
    client_dict = _FakeClient(model=_FakeRegisteredModel(aliases={}))
    assert ef.resolve_alias_version(client_dict, "cat.sch.model", "champion") is None


def test_champion_gate_present_alias_returns_version():
    """Alias present in the set -> its version (structurally, no lookup needed)."""
    client = _FakeClient(
        model=_FakeRegisteredModel(aliases=[_FakeAliasObj("champion", "7"), _FakeAliasObj("challenger", "8")])
    )
    assert ef.resolve_alias_version(client, "cat.sch.model", "champion") == "7"
    # dict-shaped aliases are also supported.
    client_dict = _FakeClient(model=_FakeRegisteredModel(aliases={"champion": 9}))
    assert ef.resolve_alias_version(client_dict, "cat.sch.model", "champion") == "9"


def test_champion_gate_unknown_alias_shape_raises_fail_closed():
    """H4 TRIPWIRE — reproduces the EXACT fail-open path: the model fetch SUCCEEDS but the alias set is
    missing / None / an unexpected shape / a malformed element. The gate must RAISE (fail closed), because it
    cannot PROVE the champion alias is absent.

    This FAILS on the pre-fix code: the old `alias_map` coerced None/unknown shapes to `{}`, so
    `resolve_alias_version` returned None ('no champion') and the caller AUTO-PROMOTED the new version over a
    possibly-live (but unreadable) champion. On the fix each of these raises instead of returning None.
    """
    # (a) aliases is None (attribute present but null) — the canonical unknown/unavailable shape.
    client_none = _FakeClient(model=_FakeRegisteredModel(aliases=None))
    with pytest.raises(ValueError):
        ef.resolve_alias_version(client_none, "cat.sch.model", "champion")

    # (b) the model object has NO `aliases` attribute at all.
    client_missing = _FakeClient(model=_FakeModelNoAliasesAttr())
    with pytest.raises(ValueError):
        ef.resolve_alias_version(client_missing, "cat.sch.model", "champion")

    # (c) an unexpected TYPE for the alias set (neither dict nor list/tuple of alias objects).
    for weird in (42, "champion=7", object()):
        client_weird = _FakeClient(model=_FakeRegisteredModel(aliases=weird))
        with pytest.raises(ValueError):
            ef.resolve_alias_version(client_weird, "cat.sch.model", "champion")

    # (d) a list whose element is NOT a recognizable alias object (missing .alias/.version) — malformed set.
    client_bad_elem = _FakeClient(model=_FakeRegisteredModel(aliases=["champion", "challenger"]))
    with pytest.raises(ValueError):
        ef.resolve_alias_version(client_bad_elem, "cat.sch.model", "champion")


def test_champion_gate_model_fetch_error_reraises_fail_closed():
    """A model-FETCH error must RE-RAISE, never be read as 'no champion' — even when its message mentions
    'alias' and 'not found' (the old free-text-regex hole). Complements the alias-shape tripwire above.
    """

    class _ModelNotFound(RuntimeError):
        pass

    # get_registered_model raises (e.g. 'alias lookup failed: model not found', permission, network) — the old
    # message-regex would have mis-read this as 'no champion'; the structural check must propagate it.
    client = _FakeClient(model_error=_ModelNotFound("alias lookup failed: model not found"))
    with pytest.raises(_ModelNotFound):
        ef.resolve_alias_version(client, "cat.sch.model", "champion")


# --- BLOCKING-3: already-escalated tickets are NOT scored at inference ---
def test_already_escalated_ticket_not_scored_at_inference():
    """An escalated ticket yields a TRAINING record (label 1, strictly-before slice) but NO inference record."""
    rows = [
        {"call_id": "c1", "customer_sentiment": "neutral", "real_time_alert": "no"},
        {"call_id": "c2", "customer_sentiment": "bad", "real_time_alert": "yes"},  # escalation fired
        {"call_id": "c3", "customer_sentiment": "bad", "real_time_alert": "no"},
    ]
    calls = make_calls(rows)

    # Training: still produced (the model must learn from pre-escalation calls).
    rec_train = ef.build_ticket_feature_record(calls, training=True)
    assert rec_train is not None
    assert rec_train["is_escalated"] == 1
    assert rec_train["num_calls_considered"] == 1  # only c1 (strictly before c2)

    # Inference: the escalation already fired -> no record (nothing to intervene on).
    rec_infer = ef.build_ticket_feature_record(calls, training=False)
    assert rec_infer is None


def test_not_yet_escalated_ticket_scored_at_inference():
    """A ticket with no 'yes' alert IS scored at inference (label 0, all calls in the window)."""
    rows = [
        {"call_id": "c1", "customer_sentiment": "neutral", "real_time_alert": "no"},
        {"call_id": "c2", "customer_sentiment": "bad", "real_time_alert": "no"},
    ]
    calls = make_calls(rows)
    rec_infer = ef.build_ticket_feature_record(calls, training=False)
    assert rec_infer is not None
    assert rec_infer["is_escalated"] == 0
    assert rec_infer["num_calls_considered"] == 2


# --- 12(c): split-by-ticket keeps all rows of a ticket in one split + is stable (item 3) ---
def test_assign_split_by_ticket_stable_and_grouped():
    ids = pd.Series([f"tkt-{i}" for i in range(2000)])

    s1 = ef.assign_split_by_ticket(ids, seed=42)
    # Stable across a reordering: same ticket -> same split regardless of row order.
    reordered = ids.sample(frac=1.0, random_state=99).reset_index(drop=True)
    s2 = ef.assign_split_by_ticket(reordered, seed=42)
    map1 = dict(zip(ids, s1))
    assert all(map1[t] == lab for t, lab in zip(reordered, s2)), "split not stable under reordering"

    # All rows of a given ticket land in ONE split (duplicate the ids and check consistency).
    dup = pd.concat([ids, ids], ignore_index=True)
    sd = ef.assign_split_by_ticket(dup, seed=42)
    dmap = {}
    for t, lab in zip(dup, sd):
        assert dmap.setdefault(t, lab) == lab, f"ticket {t} landed in >1 split"

    # No ticket appears in more than one split bucket.
    buckets = {"train": set(), "val": set(), "test": set()}
    for t, lab in zip(ids, s1):
        buckets[lab].add(t)
    assert not (buckets["train"] & buckets["val"])
    assert not (buckets["train"] & buckets["test"])
    assert not (buckets["val"] & buckets["test"])

    # Every split is non-empty and fractions are in the right ballpark for 2000 tickets.
    fracs = s1.value_counts(normalize=True)
    assert 0.15 < fracs.get("test", 0) < 0.25
    assert fracs.get("val", 0) > 0.05
    assert fracs.get("train", 0) > 0.55

    # A DIFFERENT seed yields a different assignment (the split is genuinely seeded).
    s3 = ef.assign_split_by_ticket(ids, seed=7)
    assert not s1.equals(s3)

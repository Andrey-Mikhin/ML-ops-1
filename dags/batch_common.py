import hashlib
import json
import logging
import math
import os
import re
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FEATURES = ["age", "tenure_months", "monthly_spend", "support_tickets_30d"]
LIMITS = [(18, 100), (0, 240), (0, 10000), (0, 100)]
ARTIFACTS = Path(os.environ.get("BATCH_ARTIFACTS", "/opt/airflow/include/artifacts"))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def save(value):
    content = canonical(value)
    sha = digest(content)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=ARTIFACTS, delete=False) as file:
        file.write(content)
        temporary = Path(file.name)
    temporary.replace(ARTIFACTS / f"{sha}.json")
    return {"sha256": sha}


def read(ref):
    sha = ref["sha256"]
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("Invalid artifact reference")
    content = (ARTIFACTS / f"{sha}.json").read_bytes()
    if digest(content) != sha:
        raise ValueError("Artifact checksum mismatch")
    return json.loads(content)


def business_date(explicit, interval_start):
    if explicit:
        return date.fromisoformat(explicit).isoformat()
    if interval_start is None:
        raise ValueError("business_date is required without a data interval")
    return interval_start.in_timezone("Europe/Moscow").to_date_string()

def validate_rows(rows, day):
    if not rows:
        raise ValueError("Empty rows")

    seen = set()
    required = [
        "customer_id",
        "business_date",
        "age",
        "tenure_months",
        "monthly_spend",
        "support_tickets_30d",
    ]

    for row in rows:
        for key in required:
            if key not in row:
                raise KeyError(f"Missing key: {key}")

        cid = row["customer_id"]
        if not isinstance(cid, str) or not cid:
            raise ValueError(f"Invalid customer_id: {cid!r}")
        if cid in seen:
            raise ValueError(f"Duplicate customer_id: {cid}")
        seen.add(cid)

        if row["business_date"] != day:
            raise ValueError(
                f"business_date mismatch: {row['business_date']} != {day}"
            )

        for name, (lo, hi) in zip(FEATURES, LIMITS):
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Invalid {name}: {value!r}")
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {name}: {value!r}")
            if not (lo <= value <= hi):
                raise ValueError(f"{name} out of range: {value!r}")

    return rows



def validate_model(model):
    if model["feature_names"] != FEATURES:
        raise ValueError("Feature order mismatch")
    for field in ("mean", "scale", "weights"):
        if len(model[field]) != len(FEATURES):
            raise ValueError("Model dimensions mismatch")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in model[field]):
            raise ValueError("Invalid model coefficient")
    if any(v <= 0 for v in model["scale"]) or type(model["bias"]) not in (int, float) or not math.isfinite(model["bias"]):
        raise ValueError("Invalid model scale or bias")


def predict(rows, model):
    validate_model(model)
    matrix = np.asarray([[row[name] for name in FEATURES] for row in rows], dtype=np.float64)
    normalized = (matrix - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logits = normalized @ np.asarray(model["weights"]) + model["bias"]
    scores = np.round(1 / (1 + np.exp(-np.clip(logits, -700, 700))), 8)
    return [{"customer_id": row["customer_id"], "score": float(score), "label": int(score >= 0.5)}
            for row, score in zip(rows, scores)]


def infer_batches(rows, model, batch_size):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise ValueError(f"Invalid batch_size: {batch_size!r}")
    if not (1 <= batch_size <= 1000):
        raise ValueError(f"Invalid batch_size: {batch_size!r}")

    predictions = []
    batch_index = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        batch_index += 1
        logger.info("Batch %d size %d", batch_index, len(batch))
        predictions.extend(predict(batch, model))

    return predictions


def validate_predictions(predictions, expected_ids):
    if len(predictions) != len(expected_ids):
        raise ValueError(
            f"Predictions count mismatch: {len(predictions)} != {len(expected_ids)}"
        )

    pred_ids = [p["customer_id"] for p in predictions]
    if len(pred_ids) != len(set(pred_ids)):
        raise ValueError("Duplicate customer_id in predictions")
    if set(pred_ids) != set(expected_ids):
        raise ValueError("customer_id set mismatch")

    for prediction in predictions:
        cid = prediction["customer_id"]
        score = prediction["score"]
        label = prediction["label"]

        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"Invalid score for {cid}: {score!r}")
        if not math.isfinite(score):
            raise ValueError(f"Non-finite score for {cid}: {score!r}")
        if not (0 <= score <= 1):
            raise ValueError(f"Score out of range for {cid}: {score!r}")

        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError(f"Invalid label for {cid}: {label!r}")
        if label not in (0, 1):
            raise ValueError(f"Label out of range for {cid}: {label!r}")
        if label != int(score >= 0.5):
            raise ValueError(
                f"Label mismatch for {cid}: score={score}, label={label}"
            )


def replace_day(conn, day, model_version, predictions):
    if not predictions:
        raise ValueError("Empty predictions")

    with conn.transaction():
        conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"{day}:{model_version}",),
        )

        conn.execute(
            """
            DELETE FROM predictions
            WHERE business_date = %s AND model_version = %s
            """,
            (day, model_version),
        )

        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO predictions
                    (business_date, model_version, customer_id, score, label)
                VALUES (%s, %s, %s, %s, %s)
                """,
                [
                    (day, model_version, p["customer_id"], p["score"], p["label"])
                    for p in predictions
                ],
            )




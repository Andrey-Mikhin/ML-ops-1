from datetime import timedelta

import pendulum
import requests
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.sdk import BaseHook, Param, dag, task

from batch_common import (
    business_date,
    digest,
    infer_batches,
    read,
    replace_day,
    save,
    validate_predictions,
    validate_rows,
)


@dag(
    dag_id="hw_basic_batch_inference",
    start_date=pendulum.datetime(2026, 9, 1, tz="Europe/Moscow"),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["homework", "basic"],
    params={
        "business_date": Param("2026-09-01", type="string", format="date"),
        "batch_size": Param(32, type="integer", minimum=1, maximum=1000),
    },
    default_args={
        "retries": 2,
        "retry_delay": timedelta(seconds=5),
        "execution_timeout": timedelta(minutes=2),
    },
)
def hw_basic_batch_inference():
    @task
    def extract(**context):
        params = context.get("params") or {}
        day = business_date(
            params.get("business_date"),
            context.get("data_interval_start"),
        )

        conn = BaseHook.get_connection("source_api")
        base_url = conn.host.rstrip("/")

        data_resp = requests.get(f"{base_url}/basic/{day}.json", timeout=10)
        data_resp.raise_for_status()
        data = data_resp.json()
        if data.get("business_date") != day:
            raise ValueError(
                f"business_date mismatch: expected {day}, "
                f"got {data.get('business_date')}"
            )
        rows = data["rows"]

        model_resp = requests.get(f"{base_url}/models/v1.json", timeout=10)
        model_resp.raise_for_status()
        model_bytes = model_resp.content
        model = model_resp.json()
        model_sha256 = digest(model_bytes)

        return save(
            {
                "business_date": day,
                "rows": rows,
                "model": model,
                "model_sha256": model_sha256,
            }
        )

    @task(retries=0)
    def infer(extract_ref, **context):
        payload = read(extract_ref)
        day = payload["business_date"]
        rows = payload["rows"]
        model = payload["model"]
        model_sha256 = payload["model_sha256"]

        batch_size = (context.get("params") or {})["batch_size"]

        validated_rows = validate_rows(rows, day)
        predictions = infer_batches(validated_rows, model, batch_size)
        expected_ids = [row["customer_id"] for row in validated_rows]
        validate_predictions(predictions, expected_ids)

        return save(
            {
                "business_date": day,
                "model_version": "v1",
                "predictions": predictions,
                "model_sha256": model_sha256,
            }
        )

    @task
    def publish(infer_ref):
        payload = read(infer_ref)
        day = payload["business_date"]
        model_version = payload["model_version"]
        predictions = payload["predictions"]

        hook = PostgresHook(postgres_conn_id="warehouse")
        conn = hook.get_conn()
        try:
            replace_day(conn, day, model_version, predictions)
        finally:
            conn.close()

        return {
            "business_date": day,
            "model_version": model_version,
            "rows": len(predictions),
        }

    extract_ref = extract()
    infer_ref = infer(extract_ref)
    publish(infer_ref)


dag = hw_basic_batch_inference()

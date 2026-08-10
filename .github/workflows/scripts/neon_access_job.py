#!/usr/bin/env python3

import base64
import gzip
import hashlib
import json
import math
import os
import uuid
from pathlib import Path

SCHEMA_VERSION = 1


def decode_sql(sql_gzip_base64):
    compressed = base64.b64decode(sql_gzip_base64, validate=True)
    return gzip.decompress(compressed).decode("utf-8")


def encode_value(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"type": "float", "value": str(value)}
    if isinstance(value, bytes):
        return {"type": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, list):
        return [encode_value(item) for item in value]
    if isinstance(value, tuple):
        return [encode_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): encode_value(item) for key, item in value.items()}
    return {"type": type(value).__name__, "value": str(value)}


def _base_result(*, request_id, run_id, actor, reason, sql_text):
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "run_id": str(run_id),
        "actor": actor,
        "reason": reason,
        "sql_sha256": hashlib.sha256(sql_text.encode("utf-8")).hexdigest(),
        "status": "ok",
        "results": [],
        "truncated": False,
        "error": None,
    }


def execute_request(
    driver,
    *,
    dsn,
    sql_text,
    request_id,
    run_id,
    actor,
    reason,
    max_result_bytes,
):
    result = _base_result(
        request_id=request_id,
        run_id=run_id,
        actor=actor,
        reason=reason,
        sql_text=sql_text,
    )
    returned_bytes = 0
    try:
        with driver.connect(dsn, autocommit=True, prepare_threshold=None) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql_text, prepare=False)
                for current in cursor.results():
                    columns = (
                        [column.name for column in current.description] if current.description is not None else []
                    )
                    result_set = {
                        "command_status": current.statusmessage,
                        "columns": columns,
                        "rows": [],
                    }
                    if current.description is not None:
                        for row in current:
                            if result["truncated"]:
                                continue
                            encoded_row = [encode_value(value) for value in row]
                            row_bytes = len(
                                json.dumps(
                                    encoded_row,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                    allow_nan=False,
                                ).encode("utf-8")
                            )
                            if returned_bytes + row_bytes > max_result_bytes:
                                result["truncated"] = True
                                continue
                            result_set["rows"].append(encoded_row)
                            returned_bytes += row_bytes
                    result["results"].append(result_set)
    except driver.Error as error:
        result["status"] = "error"
        result["error"] = {
            "type": type(error).__name__,
            "sqlstate": getattr(error, "sqlstate", None),
            "message": str(error),
        }
    return result


def main():
    import psycopg

    request_id = os.environ["REQUEST_ID"]
    if str(uuid.UUID(request_id)) != request_id:
        raise SystemExit("REQUEST_ID must be a canonical UUID")
    max_result_bytes = int(os.environ["MAX_RESULT_BYTES"])
    if max_result_bytes <= 0:
        raise SystemExit("MAX_RESULT_BYTES must be positive")

    sql_text = decode_sql(os.environ["SQL_GZIP_BASE64"])
    result = execute_request(
        psycopg,
        dsn=os.environ["NEON_DATABASE_URL"],
        sql_text=sql_text,
        request_id=request_id,
        run_id=os.environ["RUN_ID"],
        actor=os.environ["ACTOR"],
        reason=os.environ["REASON"],
        max_result_bytes=max_result_bytes,
    )
    result_path = Path(os.environ.get("RESULT_PATH", "result.json"))
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Neon request {request_id} finished with status={result['status']} "
        f"result_sets={len(result['results'])} truncated={result['truncated']}"
    )
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

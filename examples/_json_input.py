import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _reject_constant(value):
    raise ValueError(f"Invalid JSON numeric constant: {value}")


def read_input(description, required_keys):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True, metavar="JSON_FILE", help="JSON file containing data and parameters")
    args = parser.parse_args()
    try:
        values = json.loads(Path(args.input).read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not isinstance(values, dict):
        parser.error("The input must be a JSON object")
    missing = [key for key in required_keys if key not in values or values[key] is None]
    if missing:
        parser.error("Required values are missing: " + ", ".join(missing))
    return values


def _json_value(value):
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def print_result(result):
    print(json.dumps(_json_value(result), allow_nan=False))


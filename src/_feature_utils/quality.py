import numpy as np
import pandas as pd

from ._validation import column_names, dataframe, numeric_column, positive_scalar


def missingness_summary(table):
    dataframe(table)
    count = table.isna().sum().astype(int)
    fraction = count / len(table) if len(table) else count.astype(float) * np.nan
    return pd.DataFrame({"missing_count": count, "missing_fraction": fraction})


def fit_iqr_bounds(reference, columns, *, factor):
    names = column_names(reference, columns)
    factor = positive_scalar(factor, "factor")
    rows = []
    for name in names:
        values = numeric_column(reference[name])
        values = values[np.isfinite(values)]
        q1, q3 = np.percentile(values, [25, 75]) if values.size else (np.nan, np.nan)
        spread = q3 - q1
        rows.append(
            {
                "valid_count": int(values.size),
                "q1": float(q1),
                "q3": float(q3),
                "lower": float(q1 - factor * spread),
                "upper": float(q3 + factor * spread),
            }
        )
    return pd.DataFrame(rows, index=pd.Index(names, name="feature"))


def flag_iqr_outliers(table, bounds, *, side):
    dataframe(table)
    dataframe(bounds)
    if side not in ("both", "lower", "upper"):
        raise ValueError("side must be 'both', 'lower', or 'upper'")
    if not bounds.index.is_unique or not len(bounds):
        raise ValueError("bounds must have a nonempty unique feature index")
    column_names(bounds, ["lower", "upper"])
    names = column_names(table, bounds.index)
    lower_values = numeric_column(bounds["lower"])
    upper_values = numeric_column(bounds["upper"])
    result = pd.DataFrame(index=table.index)
    for name, lower, upper in zip(names, lower_values, upper_values):
        values = numeric_column(table[name])
        flags = pd.array([pd.NA] * len(table), dtype="boolean")
        if np.isnan(lower) and np.isnan(upper):
            result[name] = flags
            continue
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ValueError("bounds must be ordered finite values or both missing")
        valid = np.isfinite(values)
        flagged = np.zeros(len(table), dtype=bool)
        if side in ("both", "lower"):
            flagged |= values < lower
        if side in ("both", "upper"):
            flagged |= values > upper
        flags[valid] = flagged[valid]
        result[name] = flags
    return result


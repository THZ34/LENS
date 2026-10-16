import numpy as np
import pandas as pd


def real_array(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values")
    return array.astype(np.float64, copy=False)


def binary_mask(mask, ndim=None):
    array = real_array(mask, "mask")
    if array.ndim not in (2, 3) or (ndim is not None and array.ndim != ndim):
        raise ValueError("mask has an unsupported number of dimensions")
    if not np.all(np.isfinite(array)) or not np.all((array == 0) | (array == 1)):
        raise ValueError("mask must contain only boolean or binary values")
    return array.astype(bool)


def aligned_image(image, mask, name):
    array = real_array(image, name)
    if array.shape != mask.shape:
        raise ValueError(f"{name} and mask must have identical shapes")
    return array


def positive_scalar(value, name):
    array = real_array(value, name)
    if array.ndim != 0 or not np.isfinite(array) or array <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return float(array)


def dataframe(table):
    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a pandas DataFrame")
    if not table.columns.is_unique:
        raise ValueError("table columns must be unique")
    return table


def column_names(table, columns):
    dataframe(table)
    if isinstance(columns, str):
        raise ValueError("columns must be a sequence of column names")
    names = list(columns)
    if not names or len(set(names)) != len(names):
        raise ValueError("columns must be nonempty and unique")
    if any(name not in table.columns for name in names):
        raise ValueError("a requested column is missing")
    return names


def numeric_column(series):
    if (
        not pd.api.types.is_numeric_dtype(series.dtype)
        or pd.api.types.is_bool_dtype(series.dtype)
        or pd.api.types.is_complex_dtype(series.dtype)
    ):
        raise ValueError("feature columns must have real numeric dtypes")
    return series.to_numpy(dtype=float, na_value=np.nan)


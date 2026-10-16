import numpy as np

from ._validation import binary_mask, real_array


def select_label(labels, label):
    array = real_array(labels, "labels")
    value = real_array(label, "label")
    if array.ndim not in (2, 3):
        raise ValueError("labels must be a two- or three-dimensional array")
    if not np.all(np.isfinite(array)) or np.any(array < 0) or np.any(array != np.floor(array)):
        raise ValueError("labels must contain finite nonnegative integers")
    if value.ndim != 0 or not np.isfinite(value) or value < 0 or value != np.floor(value):
        raise ValueError("label must be a finite nonnegative integer")
    return array == value


def region_area(mask):
    return int(np.count_nonzero(binary_mask(mask, ndim=2)))


def region_centroid(mask):
    coordinates = np.argwhere(binary_mask(mask, ndim=2))
    if len(coordinates) == 0:
        return (float("nan"), float("nan"))
    row, column = coordinates.mean(axis=0)
    return float(row), float(column)


def circularity(area, perimeter):
    area = real_array(area, "area")
    perimeter = real_array(perimeter, "perimeter")
    if area.ndim != 0 or perimeter.ndim != 0:
        raise ValueError("area and perimeter must be scalars")
    if not np.isfinite(area) or not np.isfinite(perimeter) or area <= 0 or perimeter <= 0:
        return float("nan")
    return float(4.0 * np.pi * area / perimeter**2)


def aspect_ratio(major_axis, minor_axis):
    major = real_array(major_axis, "major_axis")
    minor = real_array(minor_axis, "minor_axis")
    if major.ndim != 0 or minor.ndim != 0:
        raise ValueError("axis lengths must be scalars")
    if not np.isfinite(major) or not np.isfinite(minor) or major <= 0 or minor <= 0:
        return float("nan")
    if major < minor:
        raise ValueError("major_axis must be greater than or equal to minor_axis")
    return float(major / minor)


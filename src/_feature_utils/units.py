import numpy as np

from ._validation import positive_scalar, real_array


def pixels_to_area_mm2(pixel_count, spacing_mm):
    counts = real_array(pixel_count, "pixel_count")
    spacing = real_array(spacing_mm, "spacing_mm")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("pixel_count must contain finite nonnegative integers")
    if spacing.shape != (2,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing_mm must contain two finite positive values")
    result = counts * spacing[0] * spacing[1]
    return float(result) if result.ndim == 0 else result


def area_to_volume_ml(slice_areas_mm2, slice_thickness_mm):
    areas = real_array(slice_areas_mm2, "slice_areas_mm2")
    thickness = positive_scalar(slice_thickness_mm, "slice_thickness_mm")
    if areas.ndim != 1 or not areas.size:
        raise ValueError("slice_areas_mm2 must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(areas)) or np.any(areas < 0):
        raise ValueError("slice areas must be finite and nonnegative")
    return float(np.sum(areas) * thickness / 1000.0)


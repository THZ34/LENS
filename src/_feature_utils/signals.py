import numpy as np

from ._validation import aligned_image, binary_mask, real_array


def intensity_statistics(image, mask, *, percentiles, ddof):
    levels = real_array(percentiles, "percentiles")
    if (
        levels.ndim != 1
        or not np.all(np.isfinite(levels))
        or np.any((levels < 0) | (levels > 100))
        or np.unique(levels).size != levels.size
    ):
        raise ValueError("percentiles must be a unique sequence of finite values from 0 to 100")
    correction = real_array(ddof, "ddof")
    if (
        correction.ndim != 0
        or not np.isfinite(correction)
        or correction < 0
        or correction != np.floor(correction)
    ):
        raise ValueError("ddof must be a finite nonnegative integer")
    correction = int(correction)
    region = binary_mask(mask)
    values = aligned_image(image, region, "image")[region]
    finite = values[np.isfinite(values)]
    keys = ("sum", "mean", "std", "min", "max", "median", "iqr")
    result = {"region_count": int(values.size), "valid_count": int(finite.size)}
    result.update({key: float("nan") for key in keys})
    result["percentiles"] = {float(level): float("nan") for level in levels}
    if not finite.size:
        return result
    p25, p50, p75 = np.percentile(finite, [25, 50, 75])
    result.update(
        sum=float(finite.sum()),
        mean=float(finite.mean()),
        std=float(finite.std(ddof=correction)) if finite.size > correction else float("nan"),
        min=float(finite.min()),
        max=float(finite.max()),
        median=float(p50),
        iqr=float(p75 - p25),
    )
    if levels.size:
        result["percentiles"] = {
            float(level): float(value)
            for level, value in zip(levels, np.percentile(finite, levels))
        }
    return result


def _paired_signals(water, fat, mask):
    region = binary_mask(mask)
    water = aligned_image(water, region, "water")
    fat = aligned_image(fat, region, "fat")
    with np.errstate(over="ignore", invalid="ignore"):
        total = water + fat
    valid = (
        region
        & np.isfinite(water)
        & np.isfinite(fat)
        & np.isfinite(total)
        & (water >= 0)
        & (fat >= 0)
        & (total > 0)
    )
    return region, water, fat, total, valid


def fat_fraction_map(water, fat, mask):
    region, water, fat, total, valid = _paired_signals(water, fat, mask)
    result = np.full(region.shape, np.nan, dtype=float)
    np.divide(fat, total, out=result, where=valid)
    return result


def fat_fraction_summary(water, fat, mask):
    region, water, fat, total, valid = _paired_signals(water, fat, mask)
    result = {
        "region_count": int(np.count_nonzero(region)),
        "valid_count": int(np.count_nonzero(valid)),
        "ratio_of_sums": float("nan"),
        "mean_fraction": float("nan"),
    }
    if np.any(valid):
        scale = float(np.max(total[valid]))
        result["ratio_of_sums"] = float(
            np.sum(fat[valid] / scale) / np.sum(total[valid] / scale)
        )
        result["mean_fraction"] = float(np.mean(fat[valid] / total[valid]))
    return result


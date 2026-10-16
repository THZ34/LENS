import numpy as np

from _json_input import print_result, read_input

from lumbar_feature_utils import (
    fat_fraction_map,
    fat_fraction_summary,
    intensity_statistics,
)


def main():
    values = read_input(
        "Calculate signal summaries from supplied inputs.",
        ("mask", "water", "fat", "percentiles", "ddof"),
    )
    water = np.asarray(values["water"], dtype=float)
    fat = np.asarray(values["fat"], dtype=float)
    mask = values["mask"]
    print_result({
        "water_statistics": intensity_statistics(
            water, mask, percentiles=values["percentiles"], ddof=values["ddof"]
        ),
        "fat_fraction_summary": fat_fraction_summary(water, fat, mask),
        "fat_fraction_map": fat_fraction_map(water, fat, mask),
    })


if __name__ == "__main__":
    main()

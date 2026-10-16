from _json_input import print_result, read_input

from lumbar_feature_utils import (
    area_to_volume_ml,
    aspect_ratio,
    circularity,
    pixels_to_area_mm2,
    region_area,
    region_centroid,
    select_label,
)


def main():
    values = read_input(
        "Calculate region measurements from supplied inputs.",
        (
            "labels",
            "label",
            "spacing_mm",
            "perimeter_pixels",
            "major_axis_pixels",
            "minor_axis_pixels",
            "slice_areas_mm2",
            "slice_thickness_mm",
        ),
    )
    mask = select_label(values["labels"], label=values["label"])
    area_pixels = region_area(mask)
    print_result({
        "area_pixels": area_pixels,
        "centroid_row_column": region_centroid(mask),
        "area_mm2": pixels_to_area_mm2(area_pixels, spacing_mm=values["spacing_mm"]),
        "circularity": circularity(area_pixels, values["perimeter_pixels"]),
        "aspect_ratio": aspect_ratio(values["major_axis_pixels"], values["minor_axis_pixels"]),
        "volume_ml": area_to_volume_ml(values["slice_areas_mm2"], values["slice_thickness_mm"]),
    })


if __name__ == "__main__":
    main()

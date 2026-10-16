import pandas as pd

from _json_input import print_result, read_input

from lumbar_feature_utils import (
    align_tables,
    fit_iqr_bounds,
    flag_iqr_outliers,
    missingness_summary,
    select_columns,
)


def main():
    values = read_input(
        "Align supplied tables and calculate quality summaries.",
        (
            "first_table",
            "second_table",
            "reference_table",
            "candidate_table",
            "id_column",
            "selected_columns",
            "feature_columns",
            "iqr_factor",
            "outlier_side",
        ),
    )
    first = pd.DataFrame(values["first_table"])
    second = pd.DataFrame(values["second_table"])
    reference = pd.DataFrame(values["reference_table"])
    candidate = pd.DataFrame(values["candidate_table"])
    for name in values["feature_columns"]:
        reference[name] = pd.to_numeric(reference[name], errors="raise")
        candidate[name] = pd.to_numeric(candidate[name], errors="raise")
    aligned = align_tables(first, second, id_column=values["id_column"])
    bounds = fit_iqr_bounds(reference, columns=values["feature_columns"], factor=values["iqr_factor"])
    flags = flag_iqr_outliers(candidate, bounds, side=values["outlier_side"])
    print_result({
        "aligned_tables": [table.to_dict(orient="records") for table in aligned],
        "selected_columns": select_columns(first, values["selected_columns"]).to_dict(orient="records"),
        "missingness": missingness_summary(first).to_dict(orient="index"),
        "bounds": bounds.to_dict(orient="index"),
        "outlier_flags": flags.to_dict(orient="records"),
    })


if __name__ == "__main__":
    main()

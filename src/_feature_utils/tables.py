from ._validation import column_names, dataframe


def select_columns(table, columns):
    names = column_names(table, columns)
    return table.loc[:, names].copy()


def _indexed_table(table, id_column):
    dataframe(table)
    if id_column not in table.columns:
        raise ValueError("the ID column is missing")
    raw_ids = table[id_column]
    if raw_ids.isna().any():
        raise ValueError("IDs must not be missing")
    ids = raw_ids.astype("string").str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("IDs must be nonempty and unique after trimming")
    result = table.copy()
    result[id_column] = ids
    return result.set_index(id_column)


def align_tables(*tables, id_column):
    if len(tables) < 2:
        raise ValueError("at least two tables are required")
    indexed = [_indexed_table(table, id_column) for table in tables]
    common = indexed[0].index
    for table in indexed[1:]:
        common = common[common.isin(table.index)]
    return tuple(table.loc[common].reset_index().copy() for table in indexed)


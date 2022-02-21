import xarray as xr
import numpy as np
import dask.dataframe as dd


def dd_agg(ddf, groupers, aggcol, aggfxn):
    """Aggregate a dask dataframe on multiple columns using a custom aggregation function,
    and split the resulting grouping column back into distinct columns."""

    if type(aggcol) == str:
        aggcol = [aggcol]
    if type(groupers) == str:
        groupers = [groupers]

    # only valid for aggregation of a single column
    assert len(aggcol) == 1

    # keep only the grouping columns and the column to be aggregated
    keep = groupers + aggcol
    drop = ddf.columns.difference(keep)

    ddf_agg = ddf.drop(drop, axis=1).groupby(groupers).agg(aggfxn).compute()

    ddf_agg['groups'] = [i[0][0] for i in ddf_agg.values]
    ddf_agg['value'] = [i[0][1][-1] for i in ddf_agg.values]

    for i, grp in enumerate(groupers):
        ddf_agg[grp] = [j[i] for j in ddf_agg['groups']]

    # remove intermediate columns
    ddf_agg = ddf_agg.drop(aggcol, axis=1).drop('groups', axis=1)

    return ddf_agg


def chunk(grouped):
    median = grouped.quantile(q=0.5)
    return median


def agg(median):
    return median


def finalize(median):
    return median


def dd_median(ddf, groupers, aggcol):
    return dd_agg(ddf=ddf, groupers=groupers, aggcol=aggcol, aggfxn=dd.Aggregation('median', chunk, agg, finalize=finalize))

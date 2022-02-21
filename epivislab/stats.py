import xarray as xr
import numpy as np
import dask.dataframe as dd


class AggStats:

    def dd_agg(self, ddf, groupers, aggcol, aggfxn):
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

    def agg(self, value):
        return value

    def finalize(self, value):
        return value


class Quantile(AggStats):

    def __init__(self, quantile):
        super(AggStats, self).__init__()
        self.quantile = quantile

    def chunk(self, grouped):
        value = grouped.quantile(q=self.quantile)
        return value

    def dd_median(self, ddf, groupers, aggcol):
        return self.dd_agg(
            ddf=ddf,
            groupers=groupers,
            aggcol=aggcol,
            aggfxn=dd.Aggregation('quantile', self.chunk, self.agg, finalize=self.finalize)
        )

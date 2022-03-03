"""Classes for calculating summary statistics for epidemic simulations
"""

import xarray as xr
import numpy as np
import dask.dataframe as dd


class AggStats:
    """Class for constructing aggregations across multiple columns in ``dask.DataFrames``
    """

    def dd_agg(self, ddf, groupers, aggcol, aggfxn):
        """Aggregate a ``dask.DataFrame`` on multiple columns using a custom aggregation function,
        and split the resulting grouping column back into distinct columns.

        Args:
            ddf (dask.DataFrame): simulation data
            aggcol (str): name of column in ``dask.DataFrame`` containing measurements to aggregate
            aggfxn (str): aggregation function name

        Returns:
            dask.DataFrame: data aggregated across groupers

        """

        if type(aggcol) == str:
            aggcol = [aggcol]
        if type(groupers) == str:
            groupers = [groupers]

        # only valid for aggregation of a single column
        assert len(aggcol) == 1

        # keep only the grouping columns and the column to be aggregated
        keep = groupers + aggcol
        drop = ddf.columns.difference(keep)

        print(f'Dropping columns {drop} and aggregating by {groupers}.')
        ddf_agg = ddf.drop(drop, axis=1).groupby(groupers).agg(aggfxn).compute()

        ddf_agg['groups'] = [i[0][0] for i in ddf_agg.values]
        ddf_agg['value'] = [i[0][1][-1] for i in ddf_agg.values]

        for i, grp in enumerate(groupers):
            ddf_agg[grp] = [j[i] for j in ddf_agg['groups']]

        # remove intermediate columns
        ddf_agg = ddf_agg.drop(aggcol, axis=1).drop('groups', axis=1)

        # for consistency with dask built-in aggregations that return dask dataframes, we'll convert
        # back to a dask dataframe with a single partition
        ddf_agg = dd.from_pandas(ddf_agg, npartitions=1)

        return ddf_agg

    def agg(self, value):
        return value

    def finalize(self, value):
        return value


class Sum(AggStats):
    """Extends :class:`AggStats` for summation aggregations.
    """

    def __init__(self):
        super(AggStats, self).__init__()

    def dd_sum(self, ddf, groupers, aggcol):
        """Passes arguments to ``AggStats.dd_agg`` for summation.

        Args:
            ddf (dask.DataFrame): simulation data
            aggcol (str): name of column in ddf containing measurements to aggregate
            aggfxn (str): name of aggregation function to use

        Returns:
            dask.DataFrame: data aggregated across groupers

        """

        ddf_sum = ddf.groupby(groupers)[aggcol].agg('sum')

        return ddf_sum


class Quantile(AggStats):
    """Extends :class:`AggStats` for quantile aggregations.

    Attributes:
        quantile (float): quantile value in the (0, 1) interval
    """

    def __init__(self, quantile):
        super(AggStats, self).__init__()
        self.quantile = quantile

    def chunk(self, grouped):
        value = grouped.quantile(q=self.quantile)
        return value

    def dd_quantile(self, ddf, groupers, aggcol):
        """Passes arguments to ``AggStats.dd_agg`` for quantile calculation.

        Args:
            ddf (dask.DataFrame): simulation data
            aggcol (str): name of column in ddf containing measurements to aggregate
            aggfxn (str): name of aggregation function to use

        Returns:
            dask.DataFrame: data aggregated across groupers

        """

        return self.dd_agg(
            ddf=ddf,
            groupers=groupers,
            aggcol=aggcol,
            aggfxn=dd.Aggregation('quantile', self.chunk, self.agg, finalize=self.finalize)
        )

import xarray as xr
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
import dask.dataframe as dd
from epivislab.stats import Median
import pytest

@pytest.fixture(params=['tests/data/test_sim_2.zarr'])
def simulation_data(request):
    d = xr.open_zarr(request.param)
    return d

class TestStats:

    def test_median(self, simulation_data):

        # calculate median in dask
        chunk_size = len(simulation_data.age) * len(simulation_data.compt) * len(simulation_data.risk) * len(simulation_data.vertex) * len(simulation_data.step)
        sims_sq = simulation_data[
            ['age', 'compt', 'index', 'risk', 'step', 'vertex', 'compt_model__state']
        ].chunk(chunks={i: chunk_size for i in simulation_data.coords}).to_dask_dataframe(
            dim_order=['compt', 'vertex', 'age', 'risk', 'step', 'index']
        )
        median = Median()
        sims_evl = median.dd_median(ddf=sims_sq, groupers=['compt', 'vertex', 'age', 'risk', 'step'],
                                    aggcol=['compt_model__state'])

        # calculate median in pandas
        sims_pd = sims_sq.compute()
        sims_pd_median = sims_pd.groupby(['compt', 'vertex', 'age', 'risk', 'step'])[
            'compt_model__state'].median().reset_index()

        # combine and check differences
        regress = pd.merge(
            sims_pd_median,
            sims_evl,
            on=['compt', 'vertex', 'age', 'risk', 'step'],
            how='outer'
        )
        regress['diff'] = regress['compt_model__state'] - regress['value']

        # check concordance between dask and pandas workflows
        assert_almost_equal(min(regress['diff']), 0.0)
        assert_almost_equal(max(regress['diff']), 0.0)

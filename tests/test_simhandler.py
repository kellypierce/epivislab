import xarray as xr
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
import dask.dataframe as dd
from epivislab.simhandler import SimHandler, EpiSummary
import pytest


@pytest.fixture(params=['tests/data/test_sim_2.zarr'])
def simulation_data(request):
    d = xr.open_zarr(request.param)
    return d


@pytest.fixture(params=range(3))
def sum_groupers(request):
    groups = [
        ['compt', 'index', 'age', 'vertex'],
        ['compt', 'index', 'risk', 'vertex'],
        ['compt', 'index', 'vertex']
    ]

    idx = request.param

    return groups[idx]


@pytest.fixture(params=range(4))
def quantile_groupers(request):
    groups = [
        ['age', 'risk', 'step', 'vertex', 'compt'],
        ['compt', 'age', 'vertex'],
        ['compt', 'risk', 'vertex'],
        ['compt', 'vertex']
    ]

    idx = request.param

    return groups[idx]


@pytest.fixture(params=[0.05, 0.25, 0.5, 0.75, 0.95])
def quantiles(request):
    return {'quantile': request.param}


class TestEpiSummary:

    def test_sum_over_groups(self, simulation_data, sum_groupers):

        sims_xr = EpiSummary(
            simulation=simulation_data,
            state_coord=['compt'],
            within_sim_coord=['age', 'risk', 'vertex'],
            time_coord=['step'],
            between_sim_coord=['index'],
            measured_coord=['compt_model__state']
        )

        grp_sum = sims_xr.sum_over_groups(sum_groupers, 'compt_model__state').compute().reset_index()

        # calculate sum in pandas
        sims_pd = sims_xr.chunk_sim.compute()
        sims_pd_sum = sims_pd.groupby(sum_groupers)['compt_model__state'].sum().reset_index()

        # combine and check differences
        regress = pd.merge(
            sims_pd_sum,
            grp_sum,
            on=sum_groupers,
            how='outer'
        )
        regress['diff'] = regress['compt_model__state_x'] - regress['compt_model__state_y']

        # check concordance between dask and pandas workflows
        assert_almost_equal(min(regress['diff']), 0.0)
        assert_almost_equal(max(regress['diff']), 0.0)

    def test_quantile_between_groups(self, simulation_data, quantile_groupers, quantiles):

        sims_xr = EpiSummary(
            simulation=simulation_data,
            state_coord=['compt'],
            within_sim_coord=['age', 'risk', 'vertex'],
            time_coord=['step'],
            between_sim_coord=['index'],
            measured_coord=['compt_model__state']
        )

        grp_q = sims_xr.quantile_between_sims(quantile_groupers, 'compt_model__state', quantiles['quantile']).compute().reset_index()

        # calculate sum, then calculate the median in pandas
        sum_before_median_groups = quantile_groupers + ['index']
        sims_pd = sims_xr.sum_over_groups(sum_before_median_groups, 'compt_model__state').compute().reset_index()
        sims_pd_quantile = sims_pd.groupby(quantile_groupers)['compt_model__state'].quantile(quantiles['quantile']).reset_index()

        # combine and check differences
        regress = pd.merge(
            sims_pd_quantile,
            grp_q,
            on=quantile_groupers,
            how='outer'
        )
        regress['diff'] = regress['compt_model__state'] - regress['value']

        # check concordance between dask and pandas workflows
        assert_almost_equal(min(regress['diff']), 0.0)
        assert_almost_equal(max(regress['diff']), 0.0)
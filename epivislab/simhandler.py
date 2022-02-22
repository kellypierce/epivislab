import xarray as xr
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask
from epivislab.stats import Sum, Quantile

class SimHandler:

    def __init__(self, simulation, within_sim_coord, between_sim_coord, measured_coord):
        self.simulation = simulation
        self.within_sim = within_sim_coord
        self.between_sim = between_sim_coord
        self.measured = measured_coord
        self.validate()
        self.make_lists()

    def validate(self):
        """Check that all coords are identified as between, within, or measured"""

        assert type(self.simulation) == dask.dataframe.core.DataFrame

        all_coords = self.within_sim + self.between_sim + self.measured
        sim_coords = self.simulation.columns
        assert len(all_coords) == len(sim_coords)

        assert set(self.measured).issubset(set(sim_coords))
        assert set(self.within_sim).issubset(set(sim_coords))
        assert set(self.between_sim).issubset(set(sim_coords))

    def make_lists(self):

        if type(self.within_sim) == str:
            self.within_sim = [self.within_sim]
        if type(self.between_sim) == str:
            self.between_sim = [self.between_sim]
        if type(self.measured) == str:
            self.measured = [self.measured]

    def sum_over_groups(self, groupers, aggcol):

        # the coordinates that separate simulations must be included as a grouping variable
        try:
            assert set(self.between_sim).issubset(set(groupers))
        except AssertionError:
            print(f'The coordinate indicating separate simulations ({self.between_sim}) must be included as a grouping variable.')
            raise AssertionError

        # the measures to aggregate must all be recognized as measurement coordinates
        try:
            assert len(set(aggcol).intersection(self.measured)) == len(aggcol)
        except AssertionError:
            print(f'Not all {aggcol} measures are listed as simulation measurements ({self.measured}).')

        print(f'Summing {aggcol} over variables {set(self.within_sim).difference(set(groupers))}; retaining groups {groupers}.')
        sum_ = Sum()
        simulation_sum = sum_.dd_sum(ddf=self.simulation, groupers=groupers, aggcol=aggcol)

        return simulation_sum

    def quantile_between_sims(self, groupers, aggcol, quantile):

        if type(aggcol) == str:
            aggcol = [aggcol]
        if type(groupers) == str:
            groupers = [groupers]

        # the coordinate that separates simulations most not be a grouping variable
        assert not set(self.between_sim).issubset(set(groupers))

        # the measures to aggregate must all be recognized as measurement coordinates
        assert len(set(aggcol).intersection(self.measured)) == len(aggcol)

        # sum over any coordinates that are not requested grouping variables
        update_groupers = set(self.within_sim).difference(groupers)
        if len(update_groupers) > 0:
            sum_cols = list(update_groupers)
            sum_cols = sum_cols + self.between_sim  # add between simulation indicator(s)
            print(f'Summing over {sum_cols} within each simulation before aggregating.')
            simulation_sum = self.sum_over_groups(groupers=list(sum_cols), aggcol=aggcol).reset_index()
            groupers = list(update_groupers)

        else:
            simulation_sum = self.simulation

        print(f'Calculating quantile {quantile} for {aggcol} after summation over variables {[None if len(update_groupers) == 0 else update_groupers]}.')
        quantile = Quantile(quantile=quantile)
        simulation_quantile = quantile.dd_quantile(ddf=simulation_sum, groupers=groupers, aggcol=aggcol)

        return simulation_quantile

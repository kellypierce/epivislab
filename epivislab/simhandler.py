import xarray as xr
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask
from epivislab.stats import Sum, Quantile

class SimHandler:

    def __init__(self, simulation, state_coord, within_sim_coord, between_sim_coord, measured_coord, time_coord):
        self.simulation = simulation
        self.state_coord = state_coord  # never sum
        self.within_sim = within_sim_coord  # only sum within simulations
        self.time_coord = time_coord  # never sum
        self.between_sim = between_sim_coord  # never sum, only summarize over simulations
        self.all_coords = None
        self.measured = measured_coord
        self.validate()
        self.make_lists()
        self.chunk_sim = self.make_chunks()

    def validate(self):
        """Check that all coords are identified as between, within, or measured"""

        assert type(self.simulation) == xr.core.dataset.Dataset

        # validate coordinates
        self.all_coords = self.within_sim + self.state_coord + self.time_coord + self.between_sim  # order is intentional
        sim_coords = [i for i in self.simulation.coords]
        assert len(self.all_coords) == len(sim_coords)
        assert set(self.within_sim).issubset(set(sim_coords))
        assert set(self.between_sim).issubset(set(sim_coords))

        # validate variables
        assert set(self.measured).issubset(set(list(self.simulation.keys())))

    def make_lists(self):

        if type(self.state_coord) == str:
            self.state_coord = [self.state_coord]
        if type(self.within_sim) == str:
            self.within_sim = [self.within_sim]
        if type(self.between_sim) == str:
            self.between_sim = [self.between_sim]
        if type(self.measured) == str:
            self.measured = [self.measured]

    def make_chunks(self):

        # get chunk size length of values in each simulation, except the measured value
        chunk_size = 0
        for coord in self.all_coords:
            chunk_size += len(self.simulation[coord])

        assert chunk_size > 0

        # strip off any values in the simulation xarray, apply chunk size, and convert to dask dataframe
        # see https://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_dask_dataframe.html for importance of
        # dimension order: "Hierarchical dimension order for the resulting dataframe. All arrays are transposed to
        # this order and then written out as flat vectors in contiguous order, so the last dimension in this list will
        # be contiguous in the resulting DataFrame. This has a major influence on which operations are efficient on
        # the resulting dask dataframe."
        simple_coords = self.all_coords + self.measured
        chunk_sim = self.simulation[simple_coords].chunk(
            chunks={i: chunk_size for i in self.all_coords}
        ).to_dask_dataframe(
            dim_order=self.all_coords
        )

        return chunk_sim

class EpiSummary(SimHandler):

    def __init__(self, simulation, state_coord, within_sim_coord, between_sim_coord, measured_coord, time_coord):
        super(SimHandler, self).__init__()
        self.simulation = simulation
        self.state_coord = state_coord
        self.within_sim = within_sim_coord
        self.time_coord = time_coord
        self.between_sim = between_sim_coord
        self.all_coords = None
        self.measured = measured_coord
        self.validate()
        self.make_lists()
        self.chunk_sim = self.make_chunks()

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
        simulation_sum = sum_.dd_sum(ddf=self.chunk_sim, groupers=groupers, aggcol=aggcol)

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
            sum_cols = groupers + self.between_sim  # add between simulation indicator(s)
            simulation_sum = self.sum_over_groups(groupers=list(sum_cols), aggcol=aggcol)

        else:
            simulation_sum = self.chunk_sim

        print(f'Calculating quantile {quantile} for {aggcol} after summation over variables {[None if len(update_groupers) == 0 else update_groupers]}.')
        quantile = Quantile(quantile=quantile)
        simulation_quantile = quantile.dd_quantile(ddf=simulation_sum, groupers=groupers, aggcol=aggcol)

        return simulation_quantile

    def prediction_interval(self, groupers, aggcol, upper, lower):

        assert upper < 1.0
        assert upper > 0.0
        assert lower < 1.0
        assert lower > 0.0
        assert upper > lower

        median = self.quantile_between_sims(groupers=groupers, aggcol=aggcol, quantile=0.5).compute().reset_index()
        upper_ = self.quantile_between_sims(groupers=groupers, aggcol=aggcol, quantile=upper).compute().reset_index()
        lower_ = self.quantile_between_sims(groupers=groupers, aggcol=aggcol, quantile=lower).compute().reset_index()

        median = median.rename(columns={'value': 'median'})
        upper_ = upper_.rename(columns={'value': 'upper'})
        lower_ = lower_.rename(columns={'value': 'lower'})

        sims_summary = pd.merge(
            pd.merge(
                upper_,
                lower_,
                on=groupers,
                how='outer'
            ),
            median,
            on=groupers,
            how='outer'
        )

        sims_ds = sims_summary.to_xarray()
        return sims_ds
"""High-level API classes for working with epidemic simulation data
"""

import xarray as xr
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask
from epivislab.stats import Sum, Quantile
from epivislab.timeseries import interval_timeseries, spaghetti_timeseries

class SimHandler:
    """Organizes ``xarray`` simulation data coordinates and manages aggregation and summary statistic calculations.

    Class instantiation automatically calls several data validation, cleaning, and organizing methods.
    See :func:`validate`, :func:`make_lists`, and :func:`chunk_sim` for more details on these methods.

    Attributes:
        simulation (xarray): simulation data
        state_coord (str, list): coodinate(s) for simulation state data (e.g. disease compartment)
        within_sim (str, list): coordinate(s) for within-simulation data
        between_sim (str, list): coordinate(s) for between-simulation data
        time_coord (str): coordinate for timestep data
    """

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
        self.chunk_sim = None
        self.make_chunks()

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
        """Convert ``state_coord``, ``within_sim``, ``between_sim``, and ``measured`` coordinates provided as
        strings to single item lists.
        """

        if type(self.state_coord) == str:
            self.state_coord = [self.state_coord]
        if type(self.within_sim) == str:
            self.within_sim = [self.within_sim]
        if type(self.between_sim) == str:
            self.between_sim = [self.between_sim]
        if type(self.measured) == str:
            self.measured = [self.measured]

    def make_chunks(self):
        """Convert the ``xarray`` :attr:`simulation` to a ``dask.DataFrame``.

        The resulting ``dask.DataFrame`` will have a chunk size equal to the length of values in each each simulation
        as inferred by the combined lengths of all simulation coordinates. The dimensions will be ordered as follows:

        - self.within_sim
        - self.state_coord
        - self.time_coord
        - self.between_sim

        Per the `Dask recommendation <https://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_dask_dataframe.html>`_.,
        the last dimension will be contiguous in the resulting ``dask.DataFrame``. This ordering ensures that replicate
        simulation measures are organized next to each other, for faster slicing and computation of between-simulation
        statistics.

        Returns:
            None; assigns chunked ``dask.DataFrame`` to :attr:`chunk_sim`
        """

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

        self.chunk_sim = chunk_sim

class EpiSummary(SimHandler):
    """Extends :class:`SimHandler` for to implement aggregations.
    """

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
        self.chunk_sim = None
        self.make_chunks()

    def sum_over_groups(self, groupers, aggcol):
        """Sum column ``aggcol`` within simulations, maintaining groups named in ``groupers``

        This method checks that ``groupers`` are not between simulation coordinates (which cannot be validly summed),
        and checks that the ``aggcol`` is a measured coordinate.

        The time and state coordinates should be explicitly listed in ``groupers``.

        Args:
            groupers (list of str): names of coordinates to maintain in aggregated data
            aggcol (str): name of measured coordinate

        Returns:
            dask.DataFrame: simulation data summed across within-simulation variables not included in ``groupers``.
        """

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
        """Calculate quantiles of column ``aggcol`` between simulations, maintaining groups named in ``groupers``

        This method checks that ``groupers`` are not between simulation coordinates (which cannot be validly summed),
        and checks that the ``aggcol`` is a measured coordinate.

        The time and state coordinates should be explicitly listed in ``groupers``. If any within-simulation coordinates
        are excluded from ``groupers``, this method will first call :func:`sum_over_groups` to sum data by the
        desired grouping. After summation is complete, valide quantiles are calculated.

        Args:
            groupers (list of str): names of coordinates to maintain in aggregated data
            aggcol (str): name of measured coordinate
            quantile (float): quantile value in the (0, 1) interval; passed to :class:`Quantile` for calculation.

        Returns:
            dask.DataFrame: simulation data quantiles by ``groupers`` calculated across all simulations.
        """

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
        """Wrapper to :func:`quantile_between_sum` to calculate upper, lower, 50% quantiles.

        Args:
            groupers (list of str): names of coordinates to maintain in aggregated data
            aggcol (str): name of measured coordinate
            upper (float): quantile value in the (0, 1) interval; passed to :func:`quantile_between_sims`
            lower (float): quantile value in the (0, 1) interval; passed to :func:`quantile_between_sims`

        Returns:
            xarray: quantile data for the grouped simulation
        """


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
        sims_summary = sims_summary.set_index(groupers)
        sims_ds = sims_summary.to_xarray()
        return sims_ds

    def interval_plot(self, groupers, aggcol, upper, lower):
        """Wrapper to prediction_interval to calculate interval and generate plot
        Args:
            groupers (list of str): names of coordinates to maintain in aggregated data
            aggcol (str): name of measured coordinate
            upper (float): quantile value in the (0, 1) interval`
            lower (float): quantile value in the (0, 1) interval`

        Returns:
            None; outputs plotly graph using plotly ``display`` method.
        """

        summary_xr = self.prediction_interval(groupers=groupers, aggcol=aggcol, upper=upper, lower=lower)

        return interval_timeseries(summary_xr=summary_xr)

    def spaghetti_plot(self, **kwargs):
        """Generate spaghetti plots directly from :attr:`simulation`

        Optionally, data can be grouped by passing ``groupers`` and ``aggcol`` arguments, which are passed on
        to :func:`timeseries.spaghetti_timeseries`.

        Args:
            **kwargs (optional): optional keyword to :func:`epivislab.timeseries.spaghetti_timeseries`

        Returns:
            None; outputs plotly graph using plotly ``display`` method.
        """

        try:
            assert len(self.between_sim) == 1

        except AssertionError:
            raise NotImplementedError('Spaghetti time series can only be created for simulations with a single between-simulation coordinate.')

        try:
            assert len(self.measured) == 1

        except AssertionError:
            raise NotImplementedError('Spaghetti time series can only be created for simulations with a single measured coordinate.')

        try:
            assert len(self.time_coord) == 1

        except AssertionError:
            raise NotImplementedError('Spaghetti time series can only be created for simulations with a single time coordinate.')

        if kwargs:
            assert 'groupers' in kwargs.keys()
            assert 'aggcol' in kwargs.keys()

            # slow, foolish and annoying: dask.DataFrame to pandas.Series w/multiindex to pandas.DataFrame to
            # pandas.DataFrame with different column names to pandas.Series w/multiindex to xarray
            # reasons: (1) no dask to xarray method yet, and (2) aggcol name is lost in dask to pandas conversion
            sum_simulation = self.sum_over_groups(groupers=kwargs['groupers'], aggcol=kwargs['aggcol']).compute().reset_index()
            sum_simulation = sum_simulation.rename(columns={'': kwargs['aggcol']})
            sum_simulation = sum_simulation.set_index(kwargs['groupers'])
            sum_simulation = sum_simulation.to_xarray()
            return spaghetti_timeseries(sum_simulation, self.time_coord[0], kwargs['aggcol'], self.between_sim[0])

        else:
            return spaghetti_timeseries(self.simulation, self.time_coord[0], self.measured[0], self.between_sim[0])



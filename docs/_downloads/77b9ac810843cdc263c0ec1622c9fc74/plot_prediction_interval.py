"""
Example prediction interval plot
"""


import xarray as xr
import plotly
from epivislab.simhandler import EpiSummary

sims = xr.open_zarr('../../../tests/data/test_sim_2.zarr/')

test = EpiSummary(
    simulation=sims,
    state_coord=['compt'],
    within_sim_coord=['age', 'risk', 'vertex'],
    time_coord=['step'],
    between_sim_coord=['index'],
    measured_coord=['compt_model__state']
)

fig = test.interval_plot(
    groupers=['age', 'risk', 'step', 'vertex', 'compt'],
    aggcol='compt_model__state',
    upper=0.9,
    lower=0.05)

plotly.io.show(fig)

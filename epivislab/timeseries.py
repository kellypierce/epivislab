import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import widgets

def get_defaults(xr):

    return {i: xr[i][0].values.item() for i in list(xr.coords) if type(xr[i][0].values) != np.datetime64}

def interval_timeseries(summary_xr):

    # select a default subset of the xarray
    defaults = get_defaults(summary_xr)
    selection = summary_xr.sel(defaults).to_dataframe().reset_index()

    # check that data for a proper interval display are present
    assert 'upper' in selection.columns
    assert 'lower' in selection.columns
    assert 'median' in selection.columns

    # define data traces
    upper_trace = go.Scatter(
        x=selection['step'],
        y=selection['upper'],
        fill=None,
        mode='lines',
        line_color='rgba(255,255,255,0.2)',
        showlegend=False,
    )

    lower_trace = go.Scatter(
        x=selection['step'],
        y=selection['lower'],
        fill='tonexty',
        mode='lines',
        fillcolor='rgba(189,0,38,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
    )

    median_trace = go.Scatter(
        x=selection['step'],
        y=selection['median'],
        line_color='rgb(255,255,255)',
        name=None,
        showlegend=False,
    )

    # build widgets
    widget_dict = {}
    for key, value_ in defaults.items():
        widget = widgets.Dropdown(
            description=str(key),
            value=value_,
            options=list(summary_xr.coords[key].values)
        )
        widget_dict[key] = widget


    def _response(change):
        selection = summary_xr.sel({key: _value.value for key, _value in widget_dict.items()}).to_dataframe().reset_index()

        with g.batch_update():
            g.data[0].y = selection['upper']
            g.data[1].y = selection['lower']
            g.data[2].y = selection['median']

    for key, val in widget_dict.items():
        val.observe(_response, names="value")

    g = go.FigureWidget(
        data=[upper_trace, lower_trace, median_trace],
        layout=go.Layout(
            plot_bgcolor='#fff',
            yaxis_title='N',
            font=dict(size=18)
        )
    )

    container1 = widgets.HBox(list(widget_dict.values()))
    display(widgets.VBox([container1, g]))
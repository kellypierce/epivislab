import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import widgets


def get_summary_defaults(xr):

    return {i: xr[i][0].values.item() for i in list(xr.coords) if type(xr[i][0].values) != np.datetime64}


def get_spaghetti_defaults(xr, index_coord):

    return {i: xr[i][0].values.item() for i in list(xr.coords) if (type(xr[i][0].values) != np.datetime64) and (i != index_coord)}


def build_widgets(data_xr, defaults):

    # build widgets
    widget_dict = {}
    for key, value_ in defaults.items():
        widget = widgets.Dropdown(
            description=str(key),
            value=value_,
            options=list(data_xr.coords[key].values)
        )
        widget_dict[key] = widget

    return widget_dict


def interval_timeseries(summary_xr):

    # select a default subset of the xarray
    defaults = get_summary_defaults(summary_xr)
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
    widget_dict = build_widgets(data_xr=summary_xr, defaults=defaults)

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


def spaghetti_timeseries(simulation_xr, x_val, y_val, index_coord):

    # select a default subset of the xarray
    defaults = get_spaghetti_defaults(simulation_xr, index_coord)

    # build widgets
    widget_dict = build_widgets(data_xr=simulation_xr, defaults=defaults)

    lines = []
    for i in simulation_xr[index_coord].values:
        defaults[index_coord] = i
        line = simulation_xr.sel(defaults).to_dataframe().reset_index()
        next_line = go.Scatter(
            x=line[x_val],
            y=line[y_val],
            fill=None,
            mode='lines',
            line_color='rgba(0, 0, 0, 0.5)',
            showlegend=False
        )
        lines.append(next_line)

    def _response(change):
        selection = simulation_xr.sel({key: _value.value for key, _value in widget_dict.items()})
        with g.batch_update():
            for plot_idx, sim_idx in enumerate(selection[index_coord]):
                line = selection.sel({index_coord: sim_idx}).to_dataframe().reset_index()
                g.data[plot_idx].y = line[y_val]

    for key, val in widget_dict.items():
        val.observe(_response, names="value")

    g = go.FigureWidget(
        data=lines,
        layout=go.Layout(
            plot_bgcolor='#fff',
            yaxis_title='N',
            font=dict(size=18)
        )
    )

    container1 = widgets.HBox(list(widget_dict.values()))
    display(widgets.VBox([container1, g]))

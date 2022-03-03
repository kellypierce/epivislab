import logging
gpd_logger = logging.getLogger('geopandas')
gpd_logger.setLevel(logging.WARNING)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
fi_logger = logging.getLogger('fiona')
fi_logger.setLevel(logging.WARNING)
import pandas as pd
import geopandas as gpd
import shapefile
import plotly.offline as po
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import xarray as xr
import xsimlab as xs
#from episimlab.partition import partition
#from episimlab.setup.coords import InitDefaultCoords
import multiprocessing as mp
from datetime import datetime


def spatial_simulation(xr_array, shape, compartment):
    xr_compt = xr_array.sel({'compartment': compartment}).sum(dim=['risk_group', 'age_group'])
    df_compt = xr_compt.to_dataframe().reset_index()
    df_compt['vertex'] = [str(int(i)) for i in df_compt['vertex']]
    df_compt['date'] = [
        datetime.strftime(pd.to_datetime(i), '%Y-%m-%d') for i in df_compt['step']
    ]
    df_shape = gpd.GeoDataFrame(
        pd.merge(df_compt, shape, left_on='vertex', right_on='GEOID10', how='left'),
        crs=shape.crs
    )
    df_shape['lon'] = [i.centroid.coords[0][0] if i else None for i in df_shape['geometry']]
    df_shape['lat'] = [i.centroid.coords[0][1] if i else None for i in df_shape['geometry']]

    return df_shape


def make_burden_plot(dataframe, start_date, stop_date, token):
    # data manipulation
    dataslice = dataframe[(dataframe['date'] >= start_date) & (dataframe['date'] <= stop_date)]
    dates = dataslice['date'].unique()

    aggregate_pct = dataframe.groupby(['date'])['apply_counts_delta__counts', 'group_pop'].sum().reset_index()
    aggregate_pct['burden_per_10k'] = (aggregate_pct['apply_counts_delta__counts'] / aggregate_pct['group_pop']) * 10000
    aggregate_pct['burden_per_10k_str'] = [str(round(i, 2)) for i in aggregate_pct['burden_per_10k']]
    agg_slice = aggregate_pct[(aggregate_pct['date'] >= start_date) & (aggregate_pct['date'] <= stop_date)]

    # templates
    hovertemplate_left = '%{customdata[0]}<br>%{customdata[1]} per 10k<extra></extra>'
    hovertemplate_right = '%{customdata[0]}<br>%{customdata[1]} per 10k<extra></extra>'

    # starting time slice
    d1 = dataslice[dataslice['date'] == min(dates)]

    # https://chart-studio.plotly.com/~empet/15243/animating-traces-in-subplotsbr/#/
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "mapbox"}, {"type": "scatter"}]],
        subplot_titles=(
        'Hospitalizations (per 10,000)\nby patient residence zip code', 'Total hospitalizations (per 10,000)')
    )
    fig.add_trace(
        go.Scattermapbox(
            lat=[d1['lat']],
            lon=[d1['lon']],
            mode='markers',
            marker=dict(
                size=25,  # d1['marker_size'],
                color=d1['burden_per_10k'],
                colorbar=dict(title='', x=0.45),
                colorscale="Viridis",
                cmax=max(d1['burden_per_10k']),
                cmin=0
            ),
            customdata=d1[['vertex', 'burden_per_10k_str']].to_numpy(),
            hovertemplate=hovertemplate_left
        ),
        row=1,
        col=1
    )

    # for some reason I don't fully understand, we need to do this twice...
    fig.add_trace(
        go.Scatter(
            x=agg_slice['date'],
            y=agg_slice['burden_per_10k'],
            mode='lines',
            line=dict(width=2, color='gray', dash='dot'),
            hoverinfo='none'
        ), row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=agg_slice['date'],
            y=agg_slice['burden_per_10k'],
            mode='lines',
            line=dict(width=2, color='gray', dash='dot'),
            hoverinfo='none'
        ), row=1, col=2
    )

    # update with successive dates
    fig.update_layout(
        mapbox=dict(
            accesstoken=token,
            bearing=0,
            center=dict(lat=30.3,
                        lon=-97.7),
            pitch=0,
            zoom=8.5,
            style='light'
        )
    )

    frames = [
        go.Frame(
            data=[
                go.Scattermapbox(
                    lat=dataslice[dataslice['date'] == dates[k]]['lat'],
                    lon=dataslice[dataslice['date'] == dates[k]]['lon'],
                    marker=dict(
                        size=dataslice[dataslice['date'] == dates[k]]['marker_size'],  # 25
                        color=dataslice[dataslice['date'] == dates[k]]['burden_per_10k'],
                        colorbar=dict(title='', x=0.45),
                        colorscale="Viridis",
                        cmax=max(dataslice['burden_per_10k']),
                        cmin=min(dataslice['burden_per_10k'])
                    ),
                    customdata=dataslice[dataslice['date'] == dates[k]][['vertex', 'burden_per_10k_str']].to_numpy(),
                    hovertemplate=hovertemplate_left
                ),
                go.Scatter(
                    x=agg_slice['date'],
                    y=agg_slice['burden_per_10k'],
                    mode='lines',
                    line=dict(width=2, color='gray')
                ),
                go.Scatter(
                    x=[dates[k]],
                    y=[sum(agg_slice[agg_slice['date'] == dates[k]]['burden_per_10k'])],
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    customdata=agg_slice[agg_slice['date'] == dates[k]][['date', 'burden_per_10k_str']].to_numpy(),
                    hovertemplate=hovertemplate_right
                )
            ],
            traces=[0, 1, 2],  # there are 2 subplots but three traces, we need to request the three traces here
            name=f'frame{k}') for k in range(len(dates))
    ]

    fig.update(frames=frames)

    sliders = [
        dict(
            steps=[
                dict(
                    method='animate',
                    args=[
                        [f'frame{k}'],
                        dict(mode='immediate', frame=dict(duration=len(dates), redraw=True),
                             transition=dict(duration=0))
                    ],
                    label=dates[k]
                ) for k in range(len(dates))
            ],
            transition=dict(duration=0),
            x=0.05,  # slider starting position
            y=-0.01,
            currentvalue=dict(
                font=dict(size=18),
                prefix='Date: ',
                visible=True,
                xanchor='left'),
            len=1.0)
    ]

    fig.update_xaxes(row=1, col=2, range=[min(dates), max(dates)])
    # fig.update_yaxes(row=1, col=2, range=[0, 142000])
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=-0.05,
                x=0.01,
                xanchor='right',
                yanchor='top',
                pad=dict(t=0, r=10),
                buttons=[
                    dict(label='Play', method='animate',
                         args=[None,
                               dict(frame=dict(duration=500, redraw=True),
                                    transition=dict(duration=100),
                                    fromcurrent=True,
                                    mode='immediate')
                               ]
                         ),
                    dict(label='Pause', method='animate',
                         args=[
                             [None],  # None must be in brackets or the pause button does not work
                             dict(frame=dict(duration=0, redraw=False),
                                  mode='immediate',
                                  transition=dict(duration=0))
                         ]
                         )
                ]
            )
        ],
        template='plotly_white',
        font=dict(size=18),
        showlegend=False,
        sliders=sliders)
    fig.update_annotations(font_size=20)

    return fig
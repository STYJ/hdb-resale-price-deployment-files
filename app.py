#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from base64 import b64encode
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dotenv import load_dotenv
import os

# Load .env variables

load_dotenv()

# Set mapbox token

token = os.environ.get("MAPBOX")
px.set_mapbox_access_token(token)

# Setup app details

app = dash.Dash(__name__)
server = app.server
app.title = 'HDB Resale Price Breakdown'

################# HELPER FUNCTIONS START #################


def round_down(num, divisor):
    return num - (num % divisor)


def round_up(num, divisor):
    return round_down(num, divisor) + divisor


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def create_df(towns, types, sizes, years):
    df = hdb_data[hdb_data.town.isin(towns)]
    df = df[df.flat_type.isin(types)]
    df = df[(df.year >= years[0]) & (df.year <= years[1])]
    df = df[(df.floor_area_sqm >= sizes[0]) & (df.floor_area_sqm <= sizes[1])]
    return df


def create_pvs_scatter_plot(df):

    # Query dataframe

    df = df.groupby(
        ['town', 'floor_area_sqm']
    ).resale_price.mean().reset_index()

    # Create figure

    fig = px.scatter(
        df,
        x='floor_area_sqm',
        y='resale_price',
        color='town',
        labels={
            'floor_area_sqm': 'Size of flat (sqm)',
            'resale_price': 'Average resale price',
            'town': 'Location'
        },
        title='Breakdown of HDB resale transactions in the last 5 years',
        height=600,
        trendline="lowess",
        hover_data={'town': False, 'floor_area_sqm': False}
    )
    fig.update_layout(hovermode='x unified')
    fig.update_traces(hovertemplate='%{y}')
    return fig


def create_map_scatter_plot(df):

    # Query dataframe

    geocode = df[['full_address', 'lat', 'lon']]
    df['psf'] = df['resale_price'] / (df['floor_area_sqm'] * 10.7639)
    df = df.groupby(['full_address', 'flat_type'])[
        ['resale_price', 'psf']].agg('mean').reset_index()
    df = df.rename(
        columns={"resale_price": "avg_resale_price", "psf": "avg_psf"})

    # Format cells

    df['avg_resale_price'] = df['avg_resale_price'].apply(human_format)
    df['avg_psf'] = df['avg_psf'].apply(human_format)

    # Add display text column

    df['display_text'] = df['flat_type'] + ": " + \
        df['avg_resale_price'] + " ($" + df['avg_psf'] + " psf)"
    df = df.groupby('full_address')['display_text'].apply(
        lambda x: "%s" % '<br>'.join(x))

    # Merge geocodes with df

    df = pd.merge(df, geocode, on="full_address")

    # Create figure

    fig = go.Figure(go.Scattermapbox(
        name="",
        mode="markers",
        lon=df['lon'],
        lat=df['lat'],
        customdata=df.display_text,
        hovertemplate="%{text}<br>" +
        "%{customdata}",
        text=df.full_address))

    fig.update_layout(
        hovermode='closest',
        height=600,
        mapbox={
            'accesstoken': token,
            'center': go.layout.mapbox.Center(
                lat=1.3521,
                lon=103.8198
            ),
            'zoom': 10},
        showlegend=False)

    return fig

################# HELPER FUNCTIONS END #################

# Read files


hdb_data = pd.read_csv(
    './dataset/2000-2020-nov-resale-price-data.csv', index_col=False)
geocode = pd.read_csv("./dataset/2000-2020-nov-geocodes.csv", index_col=False)
hdb_data = pd.merge(hdb_data, geocode, on="full_address")

# Clean data

hdb_data['resale_price'] = hdb_data['resale_price'].astype(int)
hdb_data['floor_area_sqm'] = hdb_data['floor_area_sqm'].astype(int)


# Set initial values

towns = hdb_data.town.unique()
towns.sort()

flat_types = hdb_data.flat_type.unique()
flat_types.sort()

min_size = round_down(hdb_data.floor_area_sqm.min(), 5)
max_size = round_up(hdb_data.floor_area_sqm.max(), 5)

min_year = 2015
max_year = hdb_data.year.max()

# Create initial dataframe and figures

df = create_df(towns[:5], flat_types[:5], [
               min_size, max_size], [min_year, max_year])
pvs_scatter_plot = create_pvs_scatter_plot(df)
map_scatter_plot = create_map_scatter_plot(df)

app.layout = html.Div([
    html.H1('HDB Resale Price Breakdown'),
    html.Br(),
    dcc.Dropdown(
        id='towns',
        options=[{'label': t, 'value': t} for t in towns],
        value=towns[:5],
        multi=True,
        searchable=False,
        placeholder="Select a few locations..."),
    html.Br(),
    dcc.Dropdown(
        id='types',
        options=[{'label': t, 'value': t} for t in flat_types],
        value=flat_types[:5],
        multi=True,
        searchable=False,
        placeholder="Select a few flat types..."),
    html.Br(),
    dcc.RangeSlider(
        id='sizes',
        min=min_size,
        max=max_size,
        step=5,
        marks=dict([(s, {'label': str(s)})
                    for s in list(range(min_size, max_size + 1, 5))]),
        value=[min_size, max_size]
    ),
    html.Br(),
    dcc.RangeSlider(
        id='years',
        min=min_year,
        max=max_year,
        step=1,
        marks=dict([(y, {'label': str(y)})
                    for y in list(range(min_year, max_year + 1))]),
        value=[min_year, max_year]
    ),
    html.Br(),
    dcc.Graph(id='pvs_scatter_plot', figure=pvs_scatter_plot),
    dcc.Graph(id='map_scatter_plot', figure=map_scatter_plot),


    # 2 figures
    # 4 responsive inputs
], id='container')


@app.callback([
    Output('pvs_scatter_plot', 'figure'),
    Output('map_scatter_plot', 'figure'),
], [Input('towns', 'value'),
    Input('types', 'value'),
    Input('sizes', 'value'),
    Input('years', 'value'),
    ])
def update(towns, types, sizes, years):
    if towns and types and sizes and years:
        df = create_df(towns, types, sizes, years)
        pvs_scatter_plot = create_pvs_scatter_plot(df)
        map_scatter_plot = create_map_scatter_plot(df)
        return [pvs_scatter_plot, map_scatter_plot]
    else:
        return [{}, {}]


if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=True)

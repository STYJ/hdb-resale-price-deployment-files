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

# Helper functions for cleaning data

def get_year_from_month_column(month):
    return int(month[:month.find('-')])


def add_year_of_sale_column(df):
    df['year'] = df.month.map(get_year_from_month_column)


def add_full_address_column(df):
    df['full_address'] = df['block'] + ' ' + df['street_name']
#     df.drop(columns=['block', 'street_name'], inplace=True)


def add_remaining_lease_column(df):
    df['remaining_lease'] = np.int64(99) - (df.year - df['lease_commence_date'])


def add_age_column(df):
    df['age_as_of_2021'] = 2021 - df['lease_commence_date']


def rename_columns(df):
    df['town'] = df['town'].str.title()
    df['flat_type'] = df['flat_type'].str.title()
#     df['full_address'] = df['full_address'].str.title()

# Helper function for formatting string

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


# Helper functions for creating scatterplot

def create_option(town):
    return {'label': town, 'value': town}

def create_resale_size_scatter(towns):
    if len(towns) != 0:

        # Create dataframe

        df = hdb_data[hdb_data.town.isin(towns)].groupby(
            ['town', 'floor_area_sqm']
        ).resale_price.mean().reset_index()

        # Clean dataframe

        df['resale_price'] = df['resale_price'].astype('int')
        df['floor_area_sqm'] = df['floor_area_sqm'].astype('int')

        towns = df.town.unique()

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
    else:
        return {}
    
def create_map_plot():
    
    # Create dataframe
    
    hdb_data['psf'] = hdb_data['resale_price'] / (hdb_data['floor_area_sqm'] * 10.7639)
    df = hdb_data.groupby(['full_address', 'flat_type'])[['resale_price', 'psf']].agg('mean').reset_index()
    df = df.rename(columns={"resale_price": "avg_resale_price", "psf": "avg_psf"})

    # Format cells
    
    df['avg_resale_price'] = df['avg_resale_price'].apply(human_format)
    df['avg_psf'] = df['avg_psf'].apply(human_format)

    # Add display text column
    
    df['display_text'] = df['flat_type'] + ": " + df['avg_resale_price'] + " (" + df['avg_psf'] + ")"
    df = df.groupby('full_address')['display_text'].apply(lambda x: "%s" % '<br>'.join(x))

    # Merge geocodes with df
    
    df = pd.merge(df, geocode, on="full_address")

    # Create figure
    
    fig = go.Figure(go.Scattermapbox(
        name = "",
        mode = "markers",
        lon = geocode['lon'],
        lat = geocode['lat'],
        customdata = df.display_text,
        hovertemplate =
            "Type of flat: Avg resale price (avg psf)<br>" +
            "--------------------------------------------------<br>" +
            "%{customdata}"))

    fig.update_layout(
        hovermode='closest',
        mapbox = {
            'accesstoken': token,
            'center': go.layout.mapbox.Center(
                lat=1.3521,
                lon=103.8198
            ),
            'zoom': 10},
        showlegend = False)

    return fig


# -----------------------------------------------------------------------

# Specify file paths

path_2015 = "./dataset/jan-2015-to-dec-2016.csv"
path_2017 = "./dataset/jan-2017-onwards.csv"
geocode_path = "./dataset/full_address_and_geocode.csv"

# Read files

a = pd.read_csv(path_2015)
b = pd.read_csv(path_2017)
geocode = pd.read_csv(geocode_path)
hdb_data = pd.concat([a, b])

# Cleaning the rest of the data and joining them

add_year_of_sale_column(hdb_data)
add_remaining_lease_column(hdb_data)
add_age_column(hdb_data)
add_full_address_column(hdb_data)
rename_columns(hdb_data)

# Set mapbox token

token = os.environ.get("MAPBOX")
px.set_mapbox_access_token(token)


# Create figures

scatter_plot = create_resale_size_scatter(hdb_data.town.unique())
map_plot = create_map_plot()

# Create new app

app = dash.Dash(__name__)
server = app.server
app.title = 'HDB Resale Price Breakdown'

# Render chart

app.layout = html.Div([
    html.Label('Towns'),
    dcc.Dropdown(
        id='towns',
        options=[{'label': t, 'value': t} for t in hdb_data.town.unique()],
        value=hdb_data.town.unique()[:5],
        multi=True,
        searchable=False),
    dcc.Graph(id='scatter', figure=scatter_plot),
    dcc.Graph(figure=map_plot)
    ])


@app.callback(Output('scatter', 'figure'), Input('towns', 'value'))
def update_graph(towns):
    return create_resale_size_scatter(towns)

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
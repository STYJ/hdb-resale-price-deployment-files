#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import plotly.express as px
import io
from base64 import b64encode
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


# Helper functions for cleaning data

def get_year_from_month_column(month):
    return int(month[:month.find('-')])


def add_year_of_sale_column(df):
    df['year'] = df.month.map(get_year_from_month_column)


def get_full_address(df):
    df['full_address'] = df['block'] + ' ' + df['street_name']
    df.drop(columns=['block', 'street_name'], inplace=True)


def add_remaining_lease_column(df):
    df['remaining_lease'] = np.int64(99) - (df.year - df['lease_commence_date'])


def add_age_column(df):
    df['age_as_of_2021'] = 2021 - df['lease_commence_date']


def rename_cells(df):
    df['town'] = df['town'].str.title()
    df['flat_type'] = df['flat_type'].str.title()
    df['full_address'] = df['full_address'].str.title()


# Helper functions for creating figure

def create_option(town):
    return {'label': town, 'value': town}


def create_figure(towns):
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
                'resale_price': 'Average Resale price',
                'town': 'Location'
            },
            title='Breakdown of HDB resale transactions in the last 5 years',
            height=800,
            hover_data={'floor_area_sqm': False},
            )
        fig.update_layout(hovermode='x unified')
        fig.update_traces(hovertemplate='%{y}')
        return fig
    else:
        return {}


###############################################################################

# Specify file paths

path_2015 = "./dataset/jan-2015-to-dec-2016.csv"
path_2017 = "./dataset/jan-2017-onwards.csv"

# Read files

a = pd.read_csv(path_2015)
b = pd.read_csv(path_2017)

datasets = [a, b]

# Run cleaning helper functions

for x in datasets:
    add_year_of_sale_column(x)
    add_remaining_lease_column(x)
    add_age_column(x)
    get_full_address(x)
    rename_cells(x)

# Create full dataset

hdb_data = pd.concat([a, b])

# Initialise buffer for chart

buffer = io.StringIO()

# Create default figure

fig = create_figure(hdb_data.town.unique())

# Write to buffer

fig.write_html(buffer)

html_bytes = buffer.getvalue().encode()
encoded = b64encode(html_bytes).decode()

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
        multi=True),
    dcc.Graph(id='chart', figure=fig),
    html.A(
        html.Button('Download HTML'),
        id='download',
        href='data:text/html;base64,' + encoded,
        download='plotly_graph.html')])


@app.callback(Output('chart', 'figure'), Input('towns', 'value'))
def update_graph(towns):
    return create_figure(towns)

if __name__ == '__main__':
    app.run_server(debug=True)

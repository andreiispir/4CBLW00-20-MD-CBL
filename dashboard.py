from dash import Dash, html, dcc
from datetime import datetime
import os
import pandas as pd
from dash.dependencies import Input, Output
import dash_leaflet as dl
import dash_leaflet.express as dlx
import json
from shapely import wkt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from dash import callback_context
import dash
from ILP import ilp_optimisation_model_xgboost

# Loading data set from current directory
current_dir = os.getcwd()
csv_filename = 'london_crime_with_wards.csv'
csv_path = os.path.join(current_dir, csv_filename)
df = pd.read_csv(csv_path)


# Load wards with geometry and quantile
wards_df = pd.read_csv("wards_for_map.csv")
wards_df["geometry"] = wards_df["geometry"].apply(wkt.loads)

df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')

df_filtered = df[df['Month'].dt.year > 2020]
df_filtered = df_filtered[df_filtered['Month'].dt.month.isin([11, 12, 1])]

month_map = {11: 'nov', 12: 'dec', 1: 'jan'}
df_filtered['month_col'] = df_filtered['Month'].dt.month.map(month_map)

crime_counts = df_filtered.groupby(['NAME', 'month_col']).size().unstack(fill_value=0)

for col in ['nov', 'dec', 'jan']:
    if col not in crime_counts.columns:
        crime_counts[col] = 0

wards_df = wards_df.merge(crime_counts[['nov', 'dec', 'jan']], on='NAME', how='left')

wards_df[['nov', 'dec', 'jan']] = wards_df[['nov', 'dec', 'jan']].fillna(0).astype(int)

wards_df['risky'] = wards_df[['nov', 'dec', 'jan']].sum(axis=1)

# Define custom quantile bins
quantile_bins = [0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0]
quantile_labels = [f'P{i}' for i in range(6, 0, -1)]  # P11 = top 5%, ..., P1 = bottom 20%

# Assign quantiles for each month column
wards_df['quantile_nov'] = pd.qcut(
    wards_df['nov'],
    q=quantile_bins,
    labels=quantile_labels
)

wards_df['quantile_dec'] = pd.qcut(
    wards_df['dec'],
    q=quantile_bins,
    labels=quantile_labels
)

wards_df['quantile_jan'] = pd.qcut(
    wards_df['jan'],
    q=quantile_bins,
    labels=quantile_labels
)

wards_df['quantile_risky'] = pd.qcut(
    wards_df['risky'],
    q=quantile_bins,
    labels=quantile_labels
)

# Color map for quantiles
quantile_colors = {
    "P6": "#ffffd9",
    "P5": "#d6efb3",
    "P4": "#73c8bd",
    "P3": "#2498c1",
    "P2": "#234da0",
    "P1": "#081d58"
}
# Labels for Legend 
quantile_labels_legend = {
    "P6": "Very Low",
    "P5": "Low",
    "P4": "Moderate",
    "P3": "High",
    "P2": "Very High",
    "P1": "Extremely High"
}


# Creating layers used to plot a map
def make_layer(quantile_code):
    color = quantile_colors[quantile_code]
    features = []

    for _, row in wards_df[wards_df["quantile"] == quantile_code].iterrows():
        poly = row["geometry"]
        coords = [[list(p) for p in poly.exterior.coords]] if poly.geom_type == "Polygon" else \
                 [[[list(p) for p in part.exterior.coords]] for part in poly.geoms]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": poly.geom_type,
                "coordinates": coords
            },
            "properties": {
                "NAME": row["NAME"],
                "BOROUGH": row["BOROUGH"],
                "tooltip": f"{row['NAME']} ({row['BOROUGH']})"
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    return dl.GeoJSON(
        data=geojson,
        style=dict(fillColor=color, color="black", weight=1, fillOpacity=0.6),
        zoomToBounds=False,
        zoomToBoundsOnClick=False,
        hoverStyle={"weight": 4, "color": "red"}
    )

def make_layer_quantile2(q_code, year, month):

    global wards_df
    local_wards_df = wards_df.copy()
    color = quantile_colors[q_code]
    features = []

    prediction_filename = 'ward_level_xgb_forecasts.csv'
    prediction_path = os.path.join(current_dir, prediction_filename)

    df_alicja = ilp_optimisation_model_xgboost(prediction_path, int(year), int(month))
    df_alicja = df_alicja.rename(columns={'Ward': 'NAME'})
    local_wards_df = local_wards_df.merge(df_alicja[['NAME', 'Officers']], on='NAME', how='left')

    try:
        local_wards_df['quantile2'] = pd.qcut(
            local_wards_df['Officers'],
            q=quantile_bins,
            labels=quantile_labels
        )
    except ValueError as e:
        if "Bin edges must be unique" in str(e):
            # Fallback: remove 0.8
            fallback_bins = [0, 0.15, 0.3, 0.45, 0.6, 1.0]
            fallback_labels = quantile_labels[:len(fallback_bins)-1]
            local_wards_df['quantile2'] = pd.qcut(
                local_wards_df['Officers'],
                q=fallback_bins,
                labels=fallback_labels
            )
        else:
            raise

    for _, row in local_wards_df[local_wards_df["quantile2"] == q_code].iterrows():
        poly = row["geometry"]
        coords = [[list(p) for p in poly.exterior.coords]] if poly.geom_type == "Polygon" else \
                 [[[list(p) for p in part.exterior.coords]] for part in poly.geoms]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": poly.geom_type,
                "coordinates": coords
            },
            "properties": {
                "NAME": row["NAME"],
                "BOROUGH": row["BOROUGH"],
                "tooltip": f"{row['NAME']} ({row['BOROUGH']}): {row['Officers']} officers"
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    return dl.GeoJSON(
        data=geojson,
        style=dict(fillColor=color, color="black", weight=1, fillOpacity=0.6),
        zoomToBounds=False,
        zoomToBoundsOnClick=False,
        hoverStyle={"weight": 4, "color": "red"}
    )

def make_layer_risky_months(quantile_column, q_code):
    color = quantile_colors[q_code]
    features = []

    for _, row in wards_df[wards_df[quantile_column] == q_code].iterrows():
        poly = row["geometry"]
        coords = [[list(p) for p in poly.exterior.coords]] if poly.geom_type == "Polygon" else \
                 [[[list(p) for p in part.exterior.coords]] for part in poly.geoms]
        tooltip = f"{row['NAME']} ({row['BOROUGH']})"
        features.append({
            "type": "Feature",
            "geometry": {
                "type": poly.geom_type,
                "coordinates": coords
            },
            "properties": {
                "NAME": row["NAME"],
                "BOROUGH": row["BOROUGH"],
                "tooltip": tooltip
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    return dl.GeoJSON(
        data=geojson,
        style=dict(fillColor=color, color="black", weight=1, fillOpacity=0.6),
        zoomToBounds=False,
        zoomToBoundsOnClick=False,
        hoverStyle={"weight": 4, "color": "red"}
    )


# Removing rows that don't have ward and borough names, preparing it for dropdown
def remove_non_str_rows(df, column_name):
    return df[df[column_name].apply(lambda x: isinstance(x, str))]
df_clean = remove_non_str_rows(df, 'NAME')
df_clean1 = remove_non_str_rows(df, 'BOROUGH')

# Creating sorted lists of wards and borough names that will be used for dropdown
ward_names = sorted(df_clean1['NAME'].unique())
borough_name = sorted(df_clean1['BOROUGH'].unique())

year_options = [2025, 2026, 2027]
# Creating dash app
app = Dash(__name__)
app.title = "London Police Dashboard"

# App layout
app.layout = html.Div([ 
    html.Div([
    html.H1('London Police Dashboard', style={
        'textAlign': 'center',
        'fontSize': '80px',
        'padding': '10px',
        'border': '3px solid #2f3e46',
        'borderRadius': '20px',
        'width': '60%',
        'color': '#2f3e46',
        'backgroundColor': '#84a98c'
    })
    ], style = {'display': 'flex',
        'justifyContent': 'center',
        'alignItems': 'center'}),
    html.Div([
        html.Div([
            html.Div([
            html.Label('Select year:', style = {'color': '#2f3e46', 'backgroundColor': '#84a98c', 'width': '170px',
                                                        'fontWeight': 'bold',
                                                        'marginTop': '25px',
                                                        'textAlign': 'center',
                                                        'border': '3px solid #2f3e46', 'borderRadius': '5px'}),
            dcc.RadioItems(
            id = 'year-selector',
            options = [{'label': year, 'value': year} for year in year_options],
            value = year_options[0],
            inline = True,
            labelStyle = {'marginRight': '10px', 'marginTop': '15px'},
            style = {
                'display': 'flex',
                'flex-direction': 'row',
                'align-items': 'center',
                'justify-content': 'center',
            }
            ),
            html.Label('Select month:', style={'fontWeight': 'bold', 'marginTop': '25px', 'color': '#2f3e46', 'backgroundColor': '#84a98c', 'width': '170px',
                                               'textAlign': 'center',
                                               'border': '3px solid #2f3e46', 'borderRadius': '5px'}),
            dcc.Dropdown(
                id='month-selector',
                options = [{'label': 'January', 'value': 1}, {'label': 'February', 'value': 2},
                           {'label': 'March', 'value': 3}, {'label': 'April', 'value': 4},
                           {'label': 'May', 'value': 5}, {'label': 'June', 'value': 6},
                           {'label': 'July', 'value': 7}, {'label': 'August', 'value': 8},
                           {'label': 'September', 'value': 9}, {'label': 'October', 'value': 10},
                           {'label': 'November', 'value': 11}, {'label': 'December', 'value': 12}],
                value=1,
                clearable=False,
                style={'width': '200px', 'marginTop': '5px'}
            ),
            html.Label('Search for borough:', style={'fontWeight': 'bold', 'marginTop': '30px', 'color': '#2f3e46', 'backgroundColor': '#84a98c', 'width': '200px',
                                                  'textAlign': 'center',
                                                  'border': '3px solid #2f3e46', 'borderRadius': '5px'}),
            dcc.Dropdown(id='borough-search',
                options=[{'label': name, 'value': name} for name in borough_name],
                placeholder='Select a borough...',
                searchable=True,
                clearable=True,
                style={'width': '300px', 'marginTop': '5px'}),
            html.Label('Search for ward:', style={'fontWeight': 'bold', 'marginTop': '30px', 'color': '#2f3e46', 'backgroundColor': '#84a98c', 'width': '170px',
                                                  'textAlign': 'center',
                                                  'border': '3px solid #2f3e46', 'borderRadius': '5px'}),
            dcc.Dropdown(
                id='ward-search',
                options=[{'label': name, 'value': name} for name in ward_names],
                placeholder='Select a ward...',
                searchable=True,
                clearable=True,
                style={'width': '300px', 'marginTop': '5px'}
            ),
            html.Button('Submit', id='submit-button', n_clicks=0,
                    style={'width': '100px', 'marginTop': '45px', 'marginBottom': '10px', 'color': '#2f3e46', 'backgroundColor': '#84a98c',
                           'textAlign': 'center',
                           'border': '3px solid #2f3e46', 'borderRadius': '5px',
                           'fontWeight': 'bold',
                           'fontSize': '20px'}
                    ),
            ], style = {'width': '90%',
                        'border': '3px solid #2f3e46',
                        'borderRadius': '5px',
                        'display': 'flex',
                        'flex-direction': 'column',
                        'align-items': 'center',
                        'justify-content': 'flex-start',
                        'padding': '10px',
                        'margin': '20px'
                        }),
            html.Div([
                html.Button('Display Risky Months', id='risky-months', n_clicks=0, style={'width': '300px', 'margin': '10px', 'color': '#2f3e46', 'backgroundColor': '#84a98c',
                                                                                        'textAlign': 'center',
                                                                                        'border': '3px solid #2f3e46', 'borderRadius': '5px',
                                                                                        'fontWeight': 'bold',
                                                                                        'fontSize': '20px'}),
                html.Label('Select Month:', style = {'color': '#2f3e46', 'backgroundColor': '#84a98c', 'width': '170px',
                                                        'fontWeight': 'bold',
                                                        'marginTop': '25px',
                                                        'textAlign': 'center',
                                                        'border': '3px solid #2f3e46', 'borderRadius': '5px'}),
                dcc.RadioItems(
                id = 'risky-month-selector',
                options = [{'label': 'All', 'value': 'risky'},
                           {'label': 'November', 'value': 'nov'},
                           {'label': 'December', 'value': 'dec'},
                           {'label': 'January', 'value': 'jan'}],
                value = 'risky',
                inline = True,
                labelStyle = {'marginRight': '10px', 'marginTop': '15px'},
                style = {
                    'display': 'flex',
                    'flex-direction': 'row',
                    'align-items': 'center',
                    'justify-content': 'center',
                }
                ),
            ], style = {'width': '90%',
                        'border': '3px solid #2f3e46',
                        'borderRadius': '5px',
                        'display': 'flex',
                        'flex-direction': 'column',
                        'align-items': 'center',
                        'justify-content': 'flex-start',
                        'padding': '10px',
                        'margin': '20px'
                        }),
                ], style = {
            'display': 'flex',
            'flex-direction': 'column',
            'align-items': 'center',
            'justify-content': 'flex-start',
            'padding': '10px',
            'width': '20%',
            'height': '700px',
            'fontSize': '20px',
            'margin': '20px'
        }),
        html.Div([
    dl.Map(center=[51.5074, -0.1278], zoom=11, children=[
        dl.TileLayer(url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
                     attribution='&copy; <a href="https://carto.com/">CARTO</a>'),
        #*[make_layer(q) for q in quantile_colors.keys()]
        dl.LayerGroup(id="map-layer", children=[*([make_layer(q) for q in quantile_colors.keys()])])    
    ], style={'width': '100%', 'height': '800px', 'border': '3px solid #2f3e46', 'borderRadius': '10px',}),

    html.Div(
        children=[
            html.Div("Crime Density", style={"fontWeight": "bold", "marginBottom": "10px"}),
            *[
                html.Div([
                    html.Div(style={
                        "backgroundColor": quantile_colors[q],
                        "width": "20px",
                        "height": "20px",
                        "display": "inline-block",
                        "marginRight": "10px",
                        "border": "1px solid black"
                    }),
                    html.Span(quantile_labels_legend[q])
                ], style={"marginBottom": "5px"}) for q in quantile_colors
            ]
        ],
        style={
            "position": "absolute",
            "top": "20px",
            "right": "20px",
            "zIndex": "1000",
            "backgroundColor": "white",
            "padding": "10px",
            "border": "2px solid #2f3e46",
            "borderRadius": "5px",
            "boxShadow": "2px 2px 5px rgba(0,0,0,0.3)",
            "fontSize": "13px"
        }
    )
], style={
    'position': 'relative',    
    'width': '80%',
    'height': '800px',
    'margin': '20px'
})

    ], style = {
        'marginTop': '40px',
        'display': 'flex',
        'flexDirection': 'row',
        'width': '100%'  
    }),
    html.Div([
        ####################### LUUK VISUALIZATION #########################################
        # Change style as you see fit also:
    ], style = {'width': '100%', 'height': '200px', "border": "2px solid #2f3e46", "borderRadius": "5px",}) 

], style = {'backgroundColor': '#cad2c5', 'padding': '10px', 'borderTop': '20px solid #2f3e46'})

# Update ward dropdown based on borough selection
@app.callback(
    Output('ward-search', 'options'),
    Input('ward-search', 'search_value'),
    Input('borough-search', 'value')
)
def smart_prefix_search_wards(search_value, selected_borough):
    if selected_borough:
        df_filtered = df_clean1[df_clean1['BOROUGH'] == selected_borough]
    else:
        df_filtered = df_clean

    wards_filtered = df_filtered['NAME'].unique()

    if not search_value:
        return [{'label': name, 'value': name} for name in sorted(wards_filtered)]

    search_value_lower = search_value.lower()

    def word_starts_with(text):
        return any(word.startswith(search_value_lower) for word in text.lower().split())

    matched_wards = [name for name in wards_filtered if word_starts_with(name)]

    return [{'label': name, 'value': name} for name in sorted(matched_wards)]

@app.callback(
    Output("map-layer", "children"),
    Input("submit-button", "n_clicks"),
    Input("risky-months", "n_clicks"), # Button
    Input("risky-month-selector", "value"), # Option selector
    Input('year-selector', 'value'),
    Input('month-selector', 'value'),
    prevent_initial_call=True
)
def update_map(submit_clicks, risky_clicks, risky_month, year, month):
    ctx = callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == "submit-button":
        # Original behavior: map based on quantile2 (Officers)
        return [make_layer_quantile2(q, year, month) for q in quantile_colors.keys()]

    elif trigger_id == "risky-months":
        # Behavior based on risky-month-selector
        quantile_column_map = {
            'nov': 'quantile_nov',
            'dec': 'quantile_dec',
            'jan': 'quantile_jan',
            'risky': 'quantile_risky'
        }
        quantile_column = quantile_column_map.get(risky_month, 'quantile_risky')
        return [make_layer_risky_months(quantile_column, q) for q in quantile_colors.keys()]
    #########

    else:
        raise dash.exceptions.PreventUpdate

@app.callback(
Output('borough-search', 'options'),
Input('borough-search', 'search_value')
)
def smart_prefix_search_boroughs(search_value):
    if not search_value:
        return [{'label': name, 'value': name} for name in borough_name]

    search_value_lower = search_value.lower()

    def word_starts_with(text):
        return any(word.startswith(search_value_lower) for word in text.lower().split())

    matched_boroughs = [name for name in borough_name if word_starts_with(name)]

    return [{'label': name, 'value': name} for name in sorted(matched_boroughs)]

@app.callback(
    Output('month-selector', 'options'),
    Input('year-selector', 'value')
)
def update_month_dropdown(selected_year):
    if selected_year == 2027:
        return [
            {'label': 'January', 'value': 1},
            {'label': 'February', 'value': 2}
        ]
    else:
        return [
            {'label': 'January', 'value': 1}, {'label': 'February', 'value': 2},
            {'label': 'March', 'value': 3}, {'label': 'April', 'value': 4},
            {'label': 'May', 'value': 5}, {'label': 'June', 'value': 6},
            {'label': 'July', 'value': 7}, {'label': 'August', 'value': 8},
            {'label': 'September', 'value': 9}, {'label': 'October', 'value': 10},
            {'label': 'November', 'value': 11}, {'label': 'December', 'value': 12}
        ]

if __name__ == '__main__':
    app.run(debug=True)
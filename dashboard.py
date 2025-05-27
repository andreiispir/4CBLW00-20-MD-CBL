import dash
from dash import html, dcc, Input, Output, State, ctx
import pandas as pd
import plotly.express as px
import json
import geopandas as gpd

gdf = gpd.read_file("C:/Users/mariz/OneDrive - TU Eindhoven/Desktop/statistical-gis-boundaries-london/statistical-gis-boundaries-london/ESRI/London_Ward_CityMerged.shp")
gdf.to_file("london_wards.geojson", driver="GeoJSON")

# Loading the GeoJSON
with open("london_wards.geojson", "r", encoding="utf-8") as file:
    wards_geojson = json.load(file)

df = pd.read_csv("london_crime_with_wards.csv")
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m', errors='coerce')
df['MonthYear'] = df['Month'].dt.strftime('%m/%Y')
df['BurglaryCount'] = 1

#dropdown options
date_options = [{'label': d, 'value': d} for d in sorted(df['MonthYear'].dropna().unique())]

#Dash app
app = dash.Dash(__name__)
app.title = "London Burglary Choropleth"

# Layout
app.layout = html.Div([
    html.Div([
        html.H2("Filters", style={'margin-bottom': '10px'}),

        html.Label("Select Date (MM/YYYY):"),
        dcc.Dropdown(
            id='date-dropdown',
            options=date_options,
            placeholder="Select a date",
            value=None,
            clearable=True
        ),

        html.Br(),
        html.Label("Search Ward:"),
        dcc.Input(
            id='ward-search',
            type='text',
            placeholder='Enter ward name',
            debounce=True,
            style={'width': '100%'}
        ),

        html.Br(), html.Br(),
        html.Button("Reset", id='reset-button', n_clicks=0),

    ], style={
        'width': '25%',
        'padding': '20px',
        'position': 'absolute',
        'top': '20px',
        'left': '20px',
        'backgroundColor': '#f9f9f9',
        'border': '1px solid #ddd',
        'border-radius': '5px',
        'box-shadow': '0 2px 4px rgba(0,0,0,0.1)',
        'zIndex': '999'
    }),

    html.Div([
        dcc.Graph(id='choropleth-map', style={'height': '100vh'})
    ], style={'marginLeft': '30%'})
])

# Callback
@app.callback(
    Output('choropleth-map', 'figure'),
    Output('date-dropdown', 'value'),
    Output('ward-search', 'value'),
    Input('date-dropdown', 'value'),
    Input('ward-search', 'value'),
    Input('reset-button', 'n_clicks'),
)
def update_map(selected_date, search_ward, reset_clicks):
    triggered_id = ctx.triggered_id if ctx.triggered_id else None

    # Reset
    if triggered_id == 'reset-button':
        selected_date = None
        search_ward = ''

    # Filter the data
    filtered_df = df.copy()
    if selected_date:
        filtered_df = filtered_df[filtered_df['MonthYear'] == selected_date]
    if search_ward:
        filtered_df = filtered_df[filtered_df['NAME'].str.contains(search_ward, case=False, na=False)]


    ward_counts = filtered_df.groupby('NAME').size().reset_index(name='BurglaryCount')
    all_wards = [f['properties']['NAME'] for f in wards_geojson['features']]
    ward_counts = pd.DataFrame({'NAME': all_wards}).merge(ward_counts, on='NAME', how='left').fillna(0)

    # map
    fig = px.choropleth_mapbox(
        ward_counts,
        geojson=wards_geojson,
        locations='NAME',
        featureidkey="properties.NAME",
        color='BurglaryCount',
        color_continuous_scale="Reds",
        mapbox_style="carto-positron",
        center={"lat": 51.5074, "lon": -0.1278},
        zoom=9,
        opacity=0.6,
        labels={'BurglaryCount': 'Burglary Count'}
    )
    fig.update_layout(margin={"r":0, "t":0, "l":0, "b":0})

    return fig, selected_date, search_ward


if __name__ == '__main__':
    app.run_server(debug=True)

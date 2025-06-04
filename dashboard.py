import dash
from dash import dcc, html
import geopandas as gpd
import pandas as pd
from shapely import wkt
import plotly.express as px
from dash import dcc, html, Input, Output, State

df = pd.read_csv("wards_for_map.csv")
df["geometry"] = df["geometry"].apply(wkt.loads)
geometry = gpd.GeoSeries(df["geometry"], crs="EPSG:4326")
gdf = gpd.GeoDataFrame(df, geometry=geometry)


geojson_data = gdf.reset_index().__geo_interface__

# Setup app
app = dash.Dash(__name__)
app.title = "London Ward Heatmap"

# map
def create_map(selected_names=None):
    fig = px.choropleth_mapbox(
        gdf,
        geojson=geojson_data,
        locations="NAME",
        color="no_cases",
        color_continuous_scale="YlOrRd",
        mapbox_style="carto-positron",
        zoom=9,
        center={"lat": 51.5074, "lon": -0.1278},
        opacity=0.6,
        featureidkey="properties.NAME",
        labels={"no_cases": "Number of Cases"}
    )

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        dragmode="lasso"
    )
    if selected_names:
        selected_idxs = [i for i, name in enumerate(gdf["NAME"]) if name in selected_names]
        fig.update_traces(
            selectedpoints=selected_idxs,
            selected=dict(marker=dict(opacity=1)),
            unselected=dict(marker=dict(opacity=0.3))
        )

    return fig

# Layout
app.layout = html.Div([
    html.H2("London Ward Heatmap — Burglary Counts", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Enter ward names (comma-separated):"),
        dcc.Input(id='text-input', type='text', placeholder='e.g. Willesden Green, Cricklewood', style={'width': '100%'}),
        html.Br(), html.Br(),
        html.Button("Reset Selection", id="reset-button", n_clicks=0),
    ], style={"width": "25%", "display": "inline-block", "padding": "20px", "verticalAlign": "top"}),

    html.Div([
        dcc.Graph(id="map", figure=create_map(), config={'displayModeBar': True})
    ], style={"width": "70%", "display": "inline-block", "verticalAlign": "top"})
])

# Callback
@app.callback(
    Output("map", "figure"),
    Input("map", "selectedData"),
    Input("text-input", "value"),
    Input("reset-button", "n_clicks"),
    prevent_initial_call=True
)
def update_map(selectedData, text_input, reset_clicks):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if triggered == "reset-button":
        return create_map()

    selected_names = set()

    #box/lasso selection
    if selectedData and "points" in selectedData:
        selected_names.update(p["location"] for p in selectedData["points"] if "location" in p)

    #manual text input
    if text_input:
        user_input = [name.strip() for name in text_input.split(",")]
        matched = [name for name in user_input if name in gdf["NAME"].values]
        selected_names.update(matched)

    return create_map(selected_names=selected_names if selected_names else None)


if __name__ == "__main__":
    app.run_server(debug=True)
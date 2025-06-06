from dash import Dash, html, dcc
from datetime import datetime
import os
import pandas as pd
from dash.dependencies import Input, Output
import dash_leaflet as dl
import dash_leaflet.express as dlx
from shapely import wkt

# Creating list for years supporting dates 2 years from current date
current_year = datetime.now().year
year_options = [str(current_year + i) for i in range(3)]

# Loading data set from current directory
current_dir = os.getcwd()
csv_filename = 'london_crime_with_wards.csv'
csv_path = os.path.join(current_dir, csv_filename)
df = pd.read_csv(csv_path)
# print("Crime CSV columns:", df.columns.tolist())

# Load ward-level geometry and quantile data
wards_df = pd.read_csv("wards_for_map.csv")
# print("Wards CSV columns:", wards_df.columns.tolist())
wards_df["geometry"] = wards_df["geometry"].apply(wkt.loads)

# Define quantile color scale
quantile_colors = {
    "P6": "#ffffd9",
    "P5": "#d6efb3",
    "P4": "#73c8bd",
    "P3": "#2498c1",
    "P2": "#234da0",
    "P1": "#081d58"
}

quantile_labels = {
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

# Filter helper for clean dropdown values
def remove_non_str_rows(df, column_name):
    return df[df[column_name].apply(lambda x: isinstance(x, str))]

# Use ward metadata (wards_df) to get clean dropdown options
df_clean = remove_non_str_rows(wards_df, 'NAME')
df_clean1 = remove_non_str_rows(wards_df, 'BOROUGH')
ward_names = sorted(df_clean['NAME'].unique())
borough_names = sorted(df_clean1['BOROUGH'].unique())

# Initialize Dash app
app = Dash(__name__)
app.title = "London Police Dashboard"

# Layout
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
    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}),

    html.Div([
        html.Div([
            html.Label('Select year:', style={
                'color': '#2f3e46', 'backgroundColor': '#84a98c', 'width': '170px',
                'fontWeight': 'bold', 'marginTop': '25px', 'textAlign': 'center',
                'border': '3px solid #2f3e46', 'borderRadius': '5px'}),
            dcc.RadioItems(
                id='year-selector',
                options=[{'label': year, 'value': year} for year in year_options],
                value=year_options[0],
                inline=True,
                labelStyle={'marginRight': '10px', 'marginTop': '15px'},
                style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'center', 'justify-content': 'center'}
            ),

            html.Label('Select month:', style={
                'fontWeight': 'bold', 'marginTop': '25px', 'color': '#2f3e46',
                'backgroundColor': '#84a98c', 'width': '170px', 'textAlign': 'center',
                'border': '3px solid #2f3e46', 'borderRadius': '5px'}),
            dcc.Dropdown(
                id='month-selector',
                options=[{'label': m, 'value': m} for m in [
                    'January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December'
                ]],
                value='January',
                clearable=False,
                style={'width': '200px', 'marginTop': '5px'}
            ),

            html.Label('Search for borough:', style={
                'fontWeight': 'bold', 'marginTop': '30px', 'color': '#2f3e46',
                'backgroundColor': '#84a98c', 'width': '200px', 'textAlign': 'center',
                'border': '3px solid #2f3e46', 'borderRadius': '5px'}),
            dcc.Dropdown(
                id='borough-search',
                options=[{'label': name, 'value': name} for name in borough_names],
                placeholder='Select a borough...',
                searchable=True,
                clearable=True,
                style={'width': '300px', 'marginTop': '5px'}
            ),

            html.Label('Search for ward:', style={
                'fontWeight': 'bold', 'marginTop': '30px', 'color': '#2f3e46',
                'backgroundColor': '#84a98c', 'width': '170px', 'textAlign': 'center',
                'border': '3px solid #2f3e46', 'borderRadius': '5px'}),
            dcc.Dropdown(
                id='ward-search',
                options=[{'label': name, 'value': name} for name in ward_names],
                placeholder='Select a ward...',
                searchable=True,
                clearable=True,
                style={'width': '300px', 'marginTop': '5px'}
            ),

            html.Button('Submit', id='submit-button', n_clicks=0, style={
                'width': '100px', 'marginTop': '45px', 'marginBottom': '10px',
                'color': '#2f3e46', 'backgroundColor': '#84a98c', 'textAlign': 'center',
                'border': '3px solid #2f3e46', 'borderRadius': '5px',
                'fontWeight': 'bold', 'fontSize': '20px'
            })
        ], style={
            'display': 'flex', 'flex-direction': 'column', 'align-items': 'center',
            'justify-content': 'flex-start', 'padding': '10px', 'width': '20%',
            'height': '500px', 'border': '3px solid #2f3e46', 'borderRadius': '10px',
            'fontSize': '20px', 'margin': '20px'
        }),

        html.Div([
            dl.Map(center=[51.5074, -0.1278], zoom=11, children=[
                dl.TileLayer(url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
                             attribution='&copy; <a href="https://carto.com/">CARTO</a>'),
                *[make_layer(q) for q in quantile_colors.keys()]
            ], style={'width': '100%', 'height': '800px'}),

            html.Div([
                html.Div("Crime Density", style={"fontWeight": "bold", "marginBottom": "10px"}),
                *[
                    html.Div([
                        html.Div(style={
                            "backgroundColor": quantile_colors[q],
                            "width": "20px", "height": "20px", "display": "inline-block",
                            "marginRight": "10px", "border": "1px solid black"
                        }),
                        html.Span(quantile_labels[q])
                    ], style={"marginBottom": "5px"}) for q in quantile_colors
                ]
            ], style={
                "position": "absolute", "top": "20px", "right": "20px", "zIndex": "1000",
                "backgroundColor": "white", "padding": "10px", "border": "2px solid #2f3e46",
                "borderRadius": "5px", "boxShadow": "2px 2px 5px rgba(0,0,0,0.3)", "fontSize": "13px"
            })
        ], style={
            'position': 'relative', 'width': '80%', 'height': '800px',
            'border': '3px solid #2f3e46', 'borderRadius': '10px', 'margin': '20px'
        })
    ], style={'marginTop': '40px', 'display': 'flex', 'flexDirection': 'row', 'width': '100%'})
], style={'backgroundColor': '#cad2c5', 'padding': '10px', 'borderTop': '20px solid #2f3e46'})

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
    Output('borough-search', 'options'),
    Input('borough-search', 'search_value')
)
def smart_prefix_search_boroughs(search_value):
    if not search_value:
        return [{'label': name, 'value': name} for name in borough_names]

    search_value_lower = search_value.lower()

    def word_starts_with(text):
        return any(word.startswith(search_value_lower) for word in text.lower().split())

    matched_boroughs = [name for name in borough_names if word_starts_with(name)]

    return [{'label': name, 'value': name} for name in sorted(matched_boroughs)]


# Run the app
if __name__ == '__main__':
    app.run(debug=True)

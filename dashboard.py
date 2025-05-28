from dash import Dash, html, dcc
from datetime import datetime
import os
import pandas as pd

# testing logic:
#current_year = '2028' 
#year_options = [str(int(current_year) + i) for i in range(3)]

# Creating list for years supporting dates 2 years from current date
current_year = datetime.now().year
year_options = [str(current_year + i) for i in range(3)]

# Loading data set from current directory
current_dir = os.getcwd()
csv_filename = 'london_crime_with_wards.csv'
csv_path = os.path.join(current_dir, csv_filename)
df = pd.read_csv(csv_path)

# Removing rows that don't have ward and borough names, preparing it for dropdown
def remove_non_str_rows(df, column_name):
    return df[df[column_name].apply(lambda x: isinstance(x, str))]
df_clean = remove_non_str_rows(df, 'NAME')
df_clean1 = remove_non_str_rows(df, 'BOROUGH')

# Creating sorted lists of wards and borough names that will be used for dropdown
ward_names = sorted(df_clean1['NAME'].unique())
borough_name = sorted(df_clean1['BOROUGH'].unique())

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
        html.Div([html.Label('Select year:', style = {'color': '#2f3e46', 'backgroundColor': '#84a98c', 'width': '170px',
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
                options=[{'label': month, 'value': month} for month in [
                    'January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December'
                ]],
                value='January',
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
                    )], style = {
                    'display': 'flex',
                    'flex-direction': 'column',
                    'align-items': 'center',
                    'justify-content': 'flex-start',
                    'padding': '10px',
                    'width': '20%',
                    'border': '3px solid #2f3e46',
                    'borderRadius': '10px',
                    'fontSize': '20px',
                    'margin': '20px'
        }),
        html.Div([
            html.Img(src='/assets/Figure_1.png', style={'width': '100%'})
        ], style = {
            'padding': '10px',
            'width': '80%',
            'border': '3px solid #2f3e46',
            'borderRadius': '10px',
            'margin': '20px'
        })
    ], style = {
        'marginTop': '40px',
        'display': 'flex',
        'flexDirection': 'row',
        'width': '100%'  
    })
], style = {'backgroundColor': '#cad2c5', 'padding': '10px', 'borderTop': '20px solid #2f3e46'})

if __name__ == '__main__':
    app.run(debug=True)
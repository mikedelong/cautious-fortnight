# modified from https://dash.plot.ly/live-updates
import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input
from dash.dependencies import Output
from plotly.subplots import make_subplots
from pyorbital.orbital import Orbital

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.layout = html.Div(html.Div(
    [html.H4('TERRA Satellite Live Feed'), html.Div(id='live-update-text'), dcc.Graph(id='live-update-graph'),
     dcc.Interval(id='interval-component', interval=1 * 1000, n_intervals=0)]))


@app.callback(Output('live-update-text', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_metrics(n):
    lon, lat, alt = Orbital('TERRA').get_lonlatalt(datetime.datetime.now())
    style = {'fontSize': '16px', 'padding': '5px', }
    return [
        html.Span('Longitude: {0:.2f}'.format(lon), style=style),
        html.Span('Latitude: {0:.2f}'.format(lat), style=style),
        html.Span('Altitude: {0:0.2f}'.format(alt), style=style)
    ]


# Multiple components can updated every time interval gets fired.
@app.callback(Output('live-update-graph', 'figure'), [Input('interval-component', 'n_intervals')])
def update_graph_live(n):
    data = {'time': [], 'Latitude': [], 'Longitude': [], 'Altitude': []}
    sat = Orbital('TERRA')

    # Collect some data
    for seconds in range(0, 3600, 20):
        time = datetime.datetime.now() - datetime.timedelta(seconds=seconds)
        data['time'].append(time)
        lon, lat, alt = sat.get_lonlatalt(time)
        data['Longitude'].append(lon)
        data['Latitude'].append(lat)
        data['Altitude'].append(alt)

    # Create the graph with subplots
    result = make_subplots(cols=1, rows=2, vertical_spacing=0.2, )
    result['layout']['margin'] = {'b': 30, 'l': 30, 'r': 10, 't': 10, }
    result['layout']['legend'] = {'x': 0, 'xanchor': 'left', 'y': 1, }

    result.append_trace({'mode': 'lines+markers', 'name': 'Altitude', 'type': 'scatter', 'x': data['time'],
                         'y': data['Altitude'], }, 1, 1)
    result.append_trace(
        {'mode': 'lines+markers', 'name': 'Longitude vs Latitude', 'text': data['time'], 'type': 'scatter',
         'x': data['Longitude'], 'y': data['Latitude'], }, 2, 1)

    return result


if __name__ == '__main__':
    debug = True
    app.run_server(debug=debug)

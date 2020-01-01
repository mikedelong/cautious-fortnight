# https://dash.plot.ly/dash-core-components/input
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input
from dash.dependencies import Output

app = dash.Dash(__name__)

ALLOWED_TYPES = (
    'text', 'number', 'password', 'email', 'search',
    'tel', 'url', 'range', 'hidden',
)

app.layout = html.Div(
    [dcc.Input(id='input_{}'.format(_), placeholder='input type {}'.format(_), type=_, ) for _ in ALLOWED_TYPES] + [
        html.Div(id='out-all-types')])


@app.callback(
    Output('out-all-types', 'children'),
    [Input('input_{}'.format(_), 'value') for _ in ALLOWED_TYPES],
)
def cb_render(*vals):
    return ' | '.join((str(val) for val in vals if val))


if __name__ == '__main__':
    app.run_server(debug=True)

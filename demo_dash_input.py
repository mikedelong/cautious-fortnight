# adapted from https://dash.plot.ly/dash-core-components/button
import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.layout = html.Div([html.Div(dcc.Input(id='input-box', type='text')), html.Button('Submit', id='button'),
                       html.Div(id='output-container-button', children='Enter a value and press submit'), ])

result = list()


@app.callback(
    dash.dependencies.Output('output-container-button', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')])
def update_output(n_clicks, value):
    if value:
        result.append(value)
    if n_clicks and int(n_clicks) > 0:
        return '{}'.format(result)
    else:
        return ''


if __name__ == '__main__':
    app.run_server(debug=True)

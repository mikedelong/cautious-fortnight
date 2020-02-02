from json import dumps as json_dumps
from json import load as json_load
from logging import INFO
from logging import basicConfig
from logging import getLogger
from time import time

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input
from dash.dependencies import Output
from matplotlib.pyplot import cm
from plotly.graph_objects import Figure
from plotly.graph_objects import Layout
from plotly.graph_objects import Scatter
from wordcloud import WordCloud


def float_color_to_hex(arg_float, arg_colormap):
    color_value = tuple([int(255 * arg_colormap(arg_float)[index]) for index in range(3)])
    return '#{:02x}{:02x}{:02x}'.format(color_value[0], color_value[1], color_value[2])


app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'], )
app.layout = html.Div(html.Div(
    [dcc.Graph(id='live-update-graph', figure=Figure()), html.Div(dcc.Input(id='input-box', type='text')),
     html.Button('Submit', id='button'),
     html.Div(id='output-container-button', children='Enter a value and press submit'),
     html.Div(id='intermediate-value', style={'display': 'none'})]))


@app.callback(Output('intermediate-value', 'children'), [Input('input-box', 'value')])
def process_value(value):
    if value:
        value = str(value).strip()
        if len(value) > 0:
            stop_word.append(value)
        return json_dumps(stop_word)


@app.callback(
    dash.dependencies.Output('output-container-button', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')])
def update_output(n_clicks, value):
    if n_clicks and int(n_clicks) > 0 and len(str(value).strip()):
        logger.info('new stop word is [{}]'.format(value))


@app.callback(Output('live-update-graph', 'figure'), [Input('intermediate-value', 'children')], )
def update_graph_live(n):
    count_list = sorted([this for this in list(count.items()) if this[0].lower() not in set(stop_word)],
                        key=lambda x: x[1], reverse=True, )
    to_show = {count_item[0]: count_item[1] for count_item in count_list[:token_count]}
    word_cloud = WordCloud().generate_from_frequencies(frequencies=to_show, max_font_size=max_font_size, )
    max_size = max(this[1] for this in word_cloud.layout_)
    min_size = min(this[1] for this in word_cloud.layout_)

    return Figure(data=[Scatter(mode='text', text=[this[0][0] for this in word_cloud.layout_], hoverinfo='text',
                                hovertext=['{}: {}'.format(this[0][0], count[this[0][0]], ) for this in
                                           word_cloud.layout_],
                                x=[this[2][0] for this in word_cloud.layout_],
                                y=[this[2][1] for this in word_cloud.layout_], textfont=dict(
            # todo make the sizes less disparate
            color=[float_color_to_hex(int((this[1] - min_size) * 255 / max_size), cm.get_cmap(plotly_colormap))
                   for this in word_cloud.layout_],
            size=[2 * this[1] for this in word_cloud.layout_], ))],
                  layout=Layout(autosize=True, height=800, width=1800, xaxis=dict(showticklabels=False),
                                yaxis=dict(showticklabels=False), ))


if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)
    logger.info('started.')

    with open('./afghanistan_papers.json', 'r') as settings_fp:
        settings = json_load(settings_fp, cls=None, object_hook=None, object_pairs_hook=None, parse_constant=None,
                             parse_float=None, parse_int=None, )

    dash_debug = settings['dash_debug'] if 'dash_debug' in settings.keys() else True
    if 'dash_debug' in settings.keys():
        logger.info('dash debug: {}'.format(dash_debug))
    else:
        logger.warning('dash debug not in settings; default value is {}'.format(dash_debug))
    input_file = settings['input_file'] if 'input_file' in settings.keys() else None
    if input_file:
        logger.info('input file: {}'.format(input_file))
    else:
        logger.warning('input file is missing from the settings. Quitting.')
        quit(code=1)
    max_font_size = settings['max_font_size'] if 'max_font_size' in settings.keys() else 20
    if 'max_font_size' in settings.keys():
        logger.info('max font size is {}'.format(max_font_size))
    else:
        logger.warning('max font size not in settings; using default: {}.'.format(max_font_size))
    plotly_colormap = settings['plotly_colormap'] if 'plotly_colormap' in settings.keys() else 'jet'
    if 'plotly_colormap' in settings.keys():
        logger.info('plotly/HTML colormap: {}'.format(plotly_colormap))
    else:
        logger.warning('plotly/HTML colormap not in settings; using default: {}'.format(plotly_colormap))
    stop_word = settings['stop_word'] if 'stop_word' in settings.keys() else list()
    if len(stop_word):
        with open(stop_word, 'r') as stop_word_fp:
            stop_words = json_load(stop_word_fp)
        if 'stop_word' in stop_words.keys():
            stop_word = stop_words['stop_word']
        else:
            logger.warning('stop word list malformed; check {}.'.format(settings['stop_word']))
            quit(code=4)
        logger.info('stop word list: {}'.format(stop_word))
    else:
        logger.warning('stop word list not in settings; default is empty.')
    token_count = settings['token_count'] if 'token_count' in settings.keys() else 10
    if 'token_count' in settings.keys():
        logger.info('token count: {}'.format(token_count))
    else:
        logger.warning('token count not in settings; default value is {}.'.format(token_count))

    with open(input_file, 'r') as input_fp:
        count = json_load(input_fp)

    logger.info('stop words: {}'.format(sorted(stop_word)))
    logger.info('counts: {}'.format(count))

    app.run_server(debug=dash_debug)

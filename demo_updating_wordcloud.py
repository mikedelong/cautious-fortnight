from collections import Counter
from glob import glob
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
from plotly.graph_objects import Scatter
from tika import parser
from wordcloud import WordCloud


def float_color_to_hex(arg_float, arg_colormap):
    color_value = tuple([int(255 * arg_colormap(arg_float)[index]) for index in range(3)])
    return '#{:02x}{:02x}{:02x}'.format(color_value[0], color_value[1], color_value[2])


figure = Figure()
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'], )
app.layout = html.Div(html.Div(
    [
        dcc.Graph(id='live-update-graph', figure=figure),
        html.Div(dcc.Input(id='input-box', type='text')),
        html.Button('Submit', id='button'),
        html.Div(id='output-container-button',
                 children='Enter a value and press submit'),
    ]))


@app.callback(
    dash.dependencies.Output('output-container-button', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')])
def update_output(n_clicks, value):
    if value:
        value = str(value)
        value = value.strip()
        if len(value) > 0:
            stop_word.append(value)
    if n_clicks and int(n_clicks) > 0:
        logger.info('new stop word is [{}]; stop word count is {}'.format(value, len(stop_word)))


@app.callback(Output('live-update-graph', 'figure'), [Input('input-box', 'value')], )
def update_graph_live(n):
    count_list = [this for this in list(count.items()) if this[0].lower() not in set(stop_word)]
    logger.info(count_list)
    to_show = {count_item[0]: count_item[1] for count_item in count_list[:token_count]}
    word_cloud = WordCloud().generate_from_frequencies(frequencies=to_show, max_font_size=max_font_size)
    colormap = cm.get_cmap(plotly_colormap)
    max_size = max(this[1] for this in word_cloud.layout_)
    min_size = min(this[1] for this in word_cloud.layout_)

    result = Figure(Scatter(mode='text', text=[this[0][0] for this in word_cloud.layout_],
                            x=[this[2][0] for this in word_cloud.layout_],
                            y=[this[2][1] for this in word_cloud.layout_], textfont=dict(
            color=[float_color_to_hex(int((this[1] - min_size) * 255 / max_size), colormap) for this in
                   word_cloud.layout_], size=[this[1] for this in word_cloud.layout_], )))

    return result


if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)
    logger.info('started.')

    with open('./demo_wordcloud.json', 'r') as settings_fp:
        settings = json_load(settings_fp, cls=None, object_hook=None, parse_float=None, parse_int=None,
                             parse_constant=None, object_pairs_hook=None)

    imshow_interpolation = settings['imshow_interpolation'] if 'imshow_interpolation' in settings.keys() else 20
    if 'imshow_interpolation' in settings.keys():
        logger.info('imshow interpolation: {}'.format(imshow_interpolation))
    else:
        logger.warning('imshow interpolation not in settings; default value is {}'.format(imshow_interpolation))
    input_folder = settings['input_folder'] if 'input_folder' in settings.keys() else None
    if input_folder:
        logger.info('input folder: {}'.format(input_folder))
    else:
        logger.warning('input folder is None. Quitting.')
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
        logger.info('stop word list: {}'.format(stop_word))
    else:
        logger.warning('stop word list not in settings; default is empty.')
    token_count = settings['token_count'] if 'token_count' in settings.keys() else 10
    if 'token_count' in settings.keys():
        logger.info('token count: {}'.format(token_count))
    else:
        logger.warning('token count not in settings; default value is {}.'.format(token_count))

    items = list()
    input_file_count = 0
    for input_file_index, input_file in enumerate(glob(input_folder + '*.pdf')):
        logger.info(input_file)
        input_file_count += 1
        parse_result = parser.from_file(input_file)
        items.append(parse_result['content'])

    logger.info('file count: {}'.format(input_file_count))
    logger.info('result size: {}'.format(len(items)))

    # first get all the counts
    count = Counter()
    for item_index, item in enumerate(items):
        if item is not None:
            logger.info('item: {} size: {}'.format(item_index, len(item)))
            pieces = [piece.strip() for piece in item.split()]
            pieces = [piece if not piece.endswith(':') else piece[:-1] for piece in pieces]
            pieces = [piece if not piece.endswith(';') else piece[:-1] for piece in pieces]
            pieces = [piece if not piece.endswith('.') else piece[:-1] for piece in pieces]
            pieces = [piece if not piece.endswith(',') else piece[:-1] for piece in pieces]
            for piece in pieces:
                count[piece] += 1 if all([len(piece) > 1, not piece.isdigit(), ]) else 0

    # filter out all the tokens that appear only once
    count = Counter({item: count[item] for item in count if count[item] > 1})
    logger.info('stop words: {}'.format(sorted(stop_word)))

    app.run_server(debug=True)

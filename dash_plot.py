import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import numpy as np
from scipy.signal import butter, lfilter
from board import BoardManager
import time
from brainflow.board_shim import BoardShim, BoardIds


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=-1)
    return y


board_manager = BoardManager(dev=True)
board_manager.setup_board()

sampling_rate = 125
chunk_size = 2 * sampling_rate
channel_index = 0  


app = dash.Dash(__name__)

app.layout = html.Div(
    [
        dcc.Graph(id='live-graph', animate=True),
        dcc.Interval(
            id='graph-update',
            interval=2*1000,  #ms
            n_intervals=0
        )
    ]
)

x_data = []
y_data = []

@app.callback(Output('live-graph', 'figure'),
              [Input('graph-update', 'n_intervals')])

def update_graph_live(n):
    global x_data, y_data

    data = board_manager.get_board_data()
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD)
    eeg_data = data[eeg_channels[channel_index], -chunk_size:]
    eeg_data = bandpass_filter(eeg_data, 1.0, 40.0, sampling_rate, order=5)
    current_time = [time.time() + i / sampling_rate for i in range(len(eeg_data))]
    
    x_data.extend(current_time)
    y_data.extend(eeg_data)

    data = go.Scatter(
        x=list(x_data),
        y=list(y_data),
        mode='lines',
        name='EEG Channel 1'
    )

    return {'data': [data], 'layout': go.Layout(xaxis=dict(range=[min(x_data), max(x_data)]),
                                                yaxis=dict(range=[min(y_data), max(y_data)]),
                                                title='Real-time EEG Data')}

if __name__ == '__main__':
    app.run_server(debug=True)

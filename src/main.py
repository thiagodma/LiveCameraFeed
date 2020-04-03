import dash
import dash_core_components as dcc
import dash_html_components as html
from true_value import TrueValue
from detectors import HumanDetector, FaceAgeGenderDetection
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from flask import Flask, Response
import cv2
import pandas as pd
# import pdb; pdb.set_trace()
model_path = 'faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
human_detector = HumanDetector(model_path)
agender_detector = FaceAgeGenderDetection()
truevalue = TrueValue(human_detector,agender_detector)

server = Flask(__name__)
app = dash.Dash(__name__, server=server)


class VideoCamera():
    def __init__(self):
        cap = cv2.VideoCapture()
        cap.open('http://81.14.37.24:8080/mjpg/video.mjpg?timestamp=1585844515370')
        #cap = cv2.VideoCapture('data/face-demographics-walking.mp4')
        self.video = cap

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _ , img = self.video.read()
        img = cv2.resize(img, (480, 240))
        img = truevalue.run(img)
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()


def gen(camera):
    while True:
        for _ in range(2):
            frame = camera.get_frame()
        yield(b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}


app.layout = html.Div(
    [
         html.Div(
         [
            html.H1('Webcam Test'),
            html.Img(src='/video_feed',
                     style={'height':'70%','width':'70%'})
         ], style={'textAlign':'center'}),

         html.Div(
         [
            html.Div([html.H6('Number of people in the laundry')]),

            dcc.Graph(
                id='num_people',
                figure=dict(
                        layout=dict(
                                plot_bgcolor=app_color['graph_bg'],
                                paper_bgcolor=app_color['graph_bg'])
                           ),
                     ),
            dcc.Interval(
                id='num_people_update',
                interval=10000,
                n_intervals=0
            ),
         ])

    ])

if __name__ == '__main__':
    app.run_server(debug=True)




@app.callback(
    Output('num_people', 'figure'), [Input('num_people_update', 'n_intervals')]
)
def gen_num_people(interval):
    """
    Generate the wind speed graph.
    :params interval: update the graph based on an interval
    """

    #total_time = get_current_time()
    #df = get_wind_data(total_time - 200, total_time)
    df = pd.read_csv('data.csv',index='time',parse_dates=['time']).tail(5) #gets the last 200 datapoints

    trace = dict(
        type='scatter',
        y=df['num_persons'],
        line={'color': '#42C4F7'},
        hoverinfo='skip',
        mode='lines',
    )

    layout = dict(
        plot_bgcolor=app_color['graph_bg'],
        paper_bgcolor=app_color['graph_bg'],
        font={'color': '#fff'},
        height=700,
        xaxis={
            'range': [0, 5],
            'showline': True,
            'zeroline': False,
            'fixedrange': True,
            'tickvals': [0, 1, 2, 3, 4],
            'ticktext': ['0', '1', '2', '3', '4'],
            'title': 'Time Elapsed (sec)',
        },
        yaxis={
            'range': [0,20],
            'showgrid': True,
            'showline': True,
            'fixedrange': True,
            'zeroline': False,
            'gridcolor': app_color['graph_line'],
            #'nticks': max(6, round(df['Speed'].iloc[-1] / 10)),
        },
    )

    return dict(data=[trace], layout=layout)

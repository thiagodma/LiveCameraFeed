import dash
import dash_core_components as dcc
import dash_html_components as html
from true_value import TrueValue
from datetime import datetime as dt
from detectors import HumanDetector, FaceAgeGenderDetection
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from flask import Flask, Response
import cv2,os
import pandas as pd
#model_path = 'faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
model_path = 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'

human_detector = HumanDetector(model_path,threshold=0.7)
agender_detector = FaceAgeGenderDetection()
truevalue = TrueValue(human_detector,agender_detector)
min_range = 0
max_range = 100


class VideoCamera():
    def __init__(self):
        cap = cv2.VideoCapture()
        cap.open('http://81.14.37.24:8080/mjpg/video.mjpg?timestamp=1585844515370')
        #cap = cv2.VideoCapture('data/face-demographics-walking.mp4')
        #cap = cv2.VideoCapture('data/classroom.mp4')
        #cap = cv2.VideoCapture('data/video2.mp4')
        #cap = cv2.VideoCapture('data/video1.avi')
        # cap = cv2.VideoCapture('data/TownCentreXVID.avi.1')
        self.video = cap

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _ , img = self.video.read()
        img = cv2.resize(img, (640, 400))
        try:
            img = truevalue.run(img)
        except:
            pass
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()

#generator function to send frame by frame
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
server = Flask(__name__)
app = dash.Dash(__name__, server=server,external_stylesheets=external_stylesheets)


@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}


app.layout = html.Div(
    [
         html.Div(
         [
            html.H1('Live Camera Feed'),
            html.Img(src='/video_feed',
                     style={'height':'50%','width':'50%'})
         ], style={'textAlign':'center'}),

         html.Div(
         [
            dcc.RangeSlider(
                id='range-slider',
                min=min_range,
                max=max_range,
                step=1,
                value=(min_range, max_range),
                className='six columns'
            ),

         ],className='row'),

         html.Div([html.Button('Toggle View',
                     id='button',
                     n_clicks=0),],className='row'),

         html.Div(
         [

            html.Div([
                dcc.Graph(
                    id='num_people',
                    figure=dict(
                        layout=dict(
                            plot_bgcolor=app_color["graph_bg"],
                            paper_bgcolor=app_color["graph_bg"],
                            height=400,
                            width=520,
                        )
                    ),
                ),
            ],className='six columns'),

             html.Div([
                dcc.Graph(
                    id='num_ages',
                    figure=dict(
                        layout=dict(
                            plot_bgcolor=app_color["graph_bg"],
                            paper_bgcolor=app_color["graph_bg"],
                            height=400,
                            width=520,
                        )
                    ),
                ),
             ],className='six columns'),

            dcc.Interval(
                id='update',
                interval=1000,
                n_intervals=0
            ),
         ], className='row'),
    ])


@app.callback(
    Output('num_people', 'figure'),
    [Input('update', 'n_intervals'),
     Input('button','n_clicks')],
    [State('range-slider','value')],
)
def gen_num_people(interval,n_clicks,range):
    """
    Generate the ages graph.
    interval: update the graph based on an interval
    n_clicks: number of clicks on the button
    range: user provided range
    """

    esc = dict(layout= dict(plot_bgcolor=app_color["graph_bg"],
                      paper_bgcolor=app_color["graph_bg"],
                    height=400,
                    width=520))

    if not os.path.isfile('data.csv'): return esc

    df = pd.read_csv('data.csv',index_col=0,parse_dates=['time'])
    max_range = len(df)-1

    if n_clicks%2 != 0:
        df = df.iloc[range[0]:range[1],:]
    else:
        df = df.tail(200) #gets the last 200 datapoints

    data =  [{'x':df.index,'y':df.num_people,'type':'scatter','name':'Number of people',
            'line':{'color':'#42C4F7'}, 'hoverinfo':'skip','mode':'lines'},
             {'x':df.index,'y':df.num_male,'type':'scatter','name':'Number of males',
             'line':{'color': '#F1785E'}, 'hoverinfo':'skip','mode':'lines'},
             {'x':df.index,'y':df.num_female,'type':'scatter','name':'Number of females',
             'line':{'color': '#83DD5C'}, 'hoverinfo':'skip','mode':'lines'}]

    layout = dict(
        title='Number of People',
        plot_bgcolor=app_color['graph_bg'],
        paper_bgcolor=app_color['graph_bg'],
        font={'color': '#fff'},
        height=400,
        width=520,
        xaxis={
            'showline': True,
            'zeroline': False,
            'fixedrange': True,
            'title': 'Time',
        },
        yaxis={
            'range': [0,max(df.num_people)+1],
            'showgrid': True,
            'showline': True,
            'fixedrange': True,
            'zeroline': False,
            'gridcolor': app_color['graph_line'],
        },
    )

    return dict(data=data, layout=layout)

@app.callback(
    Output('num_ages', 'figure'),
    [Input('update', 'n_intervals'),
     Input('button','n_clicks')],
    [State('range-slider','value')],
)
def gen_num_ages(interval,n_clicks,range):
    """
    Generate the ages graph.
    interval: update the graph based on an interval
    n_clicks: number of clicks on the button
    range: user provided range
    """

    esc = dict(layout= dict(plot_bgcolor=app_color['graph_bg'],
                      paper_bgcolor=app_color['graph_bg'],
                    height=400,
                    width=520))

    if not os.path.isfile('data.csv'): return esc

    df = pd.read_csv('data.csv',index_col=0,parse_dates=['time'])

    if n_clicks%2 != 0:
        df = df.iloc[range[0]:range[1],:] #if the user wants to set himself the ranges
    else:
        df = df.tail(200) #gets the last 200 datapoints

    data =  [{'x':df.index,'y':df['(0-10)'],'type':'scatter','name':'(0-10)',
            'line':{'color':'#42C4F7'}, 'hoverinfo':'skip','mode':'lines'},
             {'x':df.index,'y':df['(10-20)'],'type':'scatter','name':'(10-20)',
            'line':{'color':'#F1785E'}, 'hoverinfo':'skip','mode':'lines'},
             {'x':df.index,'y':df['(20-30)'],'type':'scatter','name':'(20-30)',
            'line':{'color':'#83DD5C'}, 'hoverinfo':'skip','mode':'lines'},
             {'x':df.index,'y':df['(30-40)'],'type':'scatter','name':'(30-40)',
            'line':{'color':'#DACE5C'}, 'hoverinfo':'skip','mode':'lines'},
             {'x':df.index,'y':df['(40-50)'],'type':'scatter','name':'(40-50)',
            'line':{'color':'#EEC5EB'}, 'hoverinfo':'skip','mode':'lines'},
             {'x':df.index,'y':df['(50-60)'],'type':'scatter','name':'(50-60)',
            'line':{'color':'#EE2BC7'}, 'hoverinfo':'skip','mode':'lines'},
             {'x':df.index,'y':df['(60-70)'],'type':'scatter','name':'(60-70)',
            'line':{'color':'#EE3A3A'}, 'hoverinfo':'skip','mode':'lines'},
             {'x':df.index,'y':df['(70-inf)'],'type':'scatter','name':'(>70)',
            'line':{'color':'#F5FD03'}, 'hoverinfo':'skip','mode':'lines'},]

    layout = dict(
        title='Number of People per Age Group',
        plot_bgcolor=app_color['graph_bg'],
        paper_bgcolor=app_color['graph_bg'],
        font={'color': '#fff'},
        height=400,
        width=520,
        xaxis={
            #'range': 10,
            'showline': True,
            'zeroline': False,
            'fixedrange': True,
            'title': 'Time',
        },
        yaxis={
            'range': [0,df.iloc[:,3:].max().max()+1],
            'showgrid': True,
            'showline': True,
            'fixedrange': True,
            'zeroline': False,
            'gridcolor': app_color['graph_line'],
        },
    )

    return dict(data=data, layout=layout)

if __name__ == '__main__':
    app.run_server(debug=True)

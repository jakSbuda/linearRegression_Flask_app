
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from joblib import load
import plotly.express as px
import plotly.graph_objects as go
import uuid


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello():
    request_type = request.method
    
    if request_type == 'GET':
        return render_template('index.html', image='static/base.svg')

    else:
        text = request.form['text']
        random_str = uuid.uuid4().hex
        path = 'static/'+ random_str +'.svg'
        np_arr = float_string_to_np(text)
        model_in = load('model.joblib')
        make_picture('AgesAndHeights.pkl',model_in, np_arr,path )

        return render_template('index.html', image=path )



def make_picture(train_data,model,new_input_np,output_file):
  df = pd.read_pickle(train_data)
  ages = df['Age']
  df = df[ages > 0]
  ages = df['Age']
  height = df['Height']

  x_new = np.array(list(range(19))).reshape(19,1)
  preds = model.predict(x_new)

  fig = px.scatter(x=ages,y=height, title="Height vs Age of People",labels={'x':'Age(years)', 'y':'Height(inches)'})

  fig.add_trace(go.Scatter(x=x_new.reshape(19),y=preds,mode='lines',name='Model'))
  new_preds = model.predict(new_input_np)

  fig.add_trace(go.Scatter(x=new_input_np.reshape(len(new_input_np)),y=new_preds, name='New Output',mode='markers', marker=dict(color='purple',size=10, line=dict(color='purple',width=2)) ))
  #.svg
  fig.write_image(output_file, width=800, engine='kaleido')
  
  fig.show()



def is_float(s):
  try:
    float(s)
    return True
  except:
    return False


def float_string_to_np(float_str):
  floats = np.array([float(x) for x in float_str.split(',') if is_float(x)])
  return floats.reshape(len(floats),1)


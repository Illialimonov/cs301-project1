from dash import dash, dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import base64
import io

df = pd.read_csv("Housing.csv").drop_duplicates()



app = dash.Dash(__name__)

app.layout = html.Div(className='main', children=[
    html.H1("Regression Model Dashboard"),
    html.Div(className = 'options', children=[
        dcc.Upload(
            id='upload-csv',
            children=html.Button('Upload CSV File'),
            multiple=False
        ),
        html.P("Select Target Variable:"),
        dcc.Dropdown(
            id='target-select',
            options = ["price", 'area', 'bedrooms', 'bathrooms', 'stories'],
            value='price',
            clearable=False,
            style = {"width": "15%"}
        ), 
        
        ]
    ),
    html.Div(className='graphs', children=[
        html.Div(className='average-by', children=[
            dcc.RadioItems(id='categorical-select', 
                           options=["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"], 
                           value="mainroad", 
                           inline=True),
            dcc.Graph(id='average-chart'),
        ]),
        
        dcc.Graph(id='corr-chart')
    ]),
    html.Div(className='train', children=[
        dcc.Checklist(id="training-features", options=df.columns.unique(), inline=True),
        dcc.Button('Train', id="train-button"),
        html.P("R2 Score:", id="r2"),
    ]),
    html.Div(className='prediction', children=[
        dcc.Input(),
        html.Button('Predict', id='predict-button'),
        html.P("Predicted value:", id="predict-value")
    ])
])

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(e)
        return False
    return True



@app.callback(
    Output('average-chart','figure'),
    Output('corr-chart','figure'),
    #Input('upload-csv', 'contents'),
    #State('upload-csv', 'filename'),
    Input('target-select', 'value'),
    Input('categorical-select', 'value')
)

def charts(target, category):
        avg_df = df.groupby(category)[target].mean().reset_index()
        corr = df.corr(numeric_only=True)[target].drop(target)
        avgChart = px.bar(
            avg_df,
            x = category,
            y = target,
            title = 'Average ' + target + ' by ' + category
        )
        corrChart = px.bar(corr)
        
        
        return avgChart, corrChart
    
#Train
@app.callback(
    Output('r2','children'),
    Input('train-button', 'n_clicks'),
    State('training-features', 'value'),
    Input('target-select', 'value'),
    
)

def train(click, features, target):
    # Convert categorical variables to numerical using one-hot encoding
    train_df = df
    yes_no_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
    train_df[yes_no_cols] = train_df[yes_no_cols].replace({"yes": 1, "no": 0})
    # Use dummies for the "furnishingstatus" column
    train_df = pd.get_dummies(train_df, columns=["furnishingstatus"], drop_first=True)

# The target variable is price, which is a numerical variable, so this is a regression problem. The dataset contains 8 features, including both numerical and categorical variables.
    y = train_df[target]   # target variable
    X = train_df[features]  # features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train a Linear Regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    #Test
    y_pred_lr = lr.predict(X_test)


    return ["R2 Score: ", r2_score(y_test, y_pred_lr)]

if __name__ == '__main__':
    app.run(debug=True)

from proj_app import app

import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie, Table
import joblib
from sqlalchemy import create_engine

# # load data v2
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load model v2
model = joblib.load("models/classifier-knn.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # extract data fpr sjpwomg percentage of data in each category (separate to 2 halves) 
    col_df = df.drop(['id','message','original','genre'],axis=1)
    y_data = list((col_df).mean())
    x_data = col_df.columns
    # Find middle index and then use it to separate data graphs
    middle_index = int(len(y_data)/2)

    # Create column names and first 20 data for table data example
    table_col_names = df.drop(['id','original','genre'],axis=1).columns[:8]
    data_df = df.drop(['id','original','genre'],axis=1)[:20]
    table_col_data = [data_df[data_1] for data_1 in table_col_names]

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    # ADD some data analysis plots to show different information about our test sets for visualisation
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=x_data[:middle_index],
                    y=y_data[:middle_index]
                )
            ],

            'layout': {
                'title': 'Percentage of classification as 1 from test dataset (first half of data)',
                'yaxis': {
                    'title': "Classification percentage (how many 1s is in the dataset)"
                },
                'xaxis': {
                    'title': "Categories"
                },
                'autosize': True

            }
        },
        {
            'data': [
                Bar(
                    x=x_data[middle_index:],
                    y=y_data[middle_index:]
                )
            ],

            'layout': {
                'title': 'Percentage of classification as 1 from test dataset (second half of data)',
                'yaxis': {
                    'title': "Classification percentage (how many 1s is in the dataset)"
                },
                'xaxis': {
                    'title': "Categories"
                },
                'autosize': True

            }
        },
        {
            'data': [
                Table(
                    header = {'values': table_col_names},
                    cells = {'values': table_col_data},
                )
            ],

            'layout': {
                'title': 'Example data set used for training in table (first 7 categories and message example)',            
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

# # Uncomment to be able to python run.py
# def main():
#     app.run(host='0.0.0.0', port=3001, debug=True)

# if __name__ == '__main__':
#     main()
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    df_sums = df.copy()
    df_sums.drop(labels=["id","message","genre","original"], axis = 1, inplace=True)
    col_sums = df_sums.sum()
    column_names = [name.capitalize().replace("_", " ") for name in col_sums.index]
    totals = list(col_sums)
    
    df_filtered = df_sums[['water','food','shelter']]
    col_sum2 = df_filtered.sum()
    totals2 = list(col_sum2)
    column_names2 = [name.capitalize() for name in col_sums.index]
    
    
    # create visuals
    
    graphs = [
        {
            'data': [
                Bar(
                    x=column_names,
                    y=totals
                )
            ],

            'layout': {
                'title': 'Distribution of Message Types',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message types",
                    'title_standoff': 200
                    
                },
                'height': 600,
                'margin': {
                    'b': 150  
                }
            }
        },
        {
            'data': [
                Bar(
                    x=column_names2,
                    y=totals2
                )
            ],

            'layout': {
                'title': 'Key Messages: Water, Shelter, and Food',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message type",
                    'title_standoff': 200
                    
                },
                'height': 400,
                'width': 500,
                'margin': {
                    'b': 50  
                }
            }
        }
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


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
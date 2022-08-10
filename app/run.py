import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
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
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    direct_c = df[df.genre == 'direct']
    direct_c_cnt = (direct_c.mean()*direct_c.shape[0])
    direct_c_cnt = direct_c_cnt.sort_values(ascending=False)[1:]
    direct_c_n = list(direct_c_cnt.index)

    news_c = df[df.genre == 'news']
    news_c_cnt = (news_c.mean()*news_c.shape[0])
    news_c_cnt = news_c_cnt.sort_values(ascending=False)[1:]
    news_c_n = list(news_c_cnt.index)

    social_c = df[df.genre == 'social']
    social_c_cnt = (social_c.mean()*social_c.shape[0])
    social_c_cnt = social_c_cnt.sort_values(ascending=False)[1:]
    social_c_n = list(social_c_cnt.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
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
                    x=direct_c_n,
                    y=direct_c_cnt
                )
            ],

            'layout': {
                'title': 'Distribution of Categories in Direct Type',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories Direct"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=social_c_n,
                    y=social_c_cnt
                )
            ],

            'layout': {
                'title': 'Distribution of Categories in Social Type',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories Social"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=news_c_n,
                    y=news_c_cnt
                )
            ],

            'layout': {
                'title': 'Distribution of Categories in News Type',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories News"
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
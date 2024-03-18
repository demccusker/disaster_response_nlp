import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import re
import numpy as np
import pandas as pd
import sqlite3
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV



def load_data(database_filepath):
    '''
    Loads data to run NLP pipeline on
    inputs: database filepath
    outputs: two dataframes, X and Y, that contain the messages and categories respectively
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(database_filepath,engine)
    X = df['message']
    Y = df.drop(["id","message","original","genre"],axis=1)
    category_names = df.columns
    return X, Y, category_names

def tokenize(text):
    '''
    Takes in raw messages and seperates, lemmatizes, and standardizes text
    inputs: text
    outputs: cleaned tokens
    '''
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
    # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Builds the model by building a pipeline, setting paramaters, and running a grid search. Returns the model with best parameters
    inputs: none
    outputs: model 
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier())),
    ])
    parameters = {
    'vect__ngram_range': ((1, 1), (1, 2)),
    'tfidf__use_idf': (True, False),
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the fit of the model on the test data. Shows precision, recall, f1-score, and 
    support for each column
    inputs: the model, x test and y test data sets, and list of category names
    returns: none
    outputs: printed report of precision, recall, f1-score, and support for each column 
    and best parameter set from the grid search
    '''
    predicted = model.predict(X_test)
    for i, column in enumerate(Y_test.columns):
        print(f"Classification Report for column: {column}")
        print(classification_report(Y_test.iloc[:, i], predicted[:, i]))
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    '''
    Saves model to a pickle file
    inputs: model and file path to save the model to
    returns: none
    outputs:  saves the model at the specified filepath
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
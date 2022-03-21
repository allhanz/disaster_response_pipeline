# import libraries
import os
import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

import warnings
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

import string
nltk.download(['punkt', 'wordnet', 'stopwords'])
warnings.simplefilter('ignore')


def load_data(database_filepath="disasterData.db"):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("disasterData", engine)
    #in order to solve the Error for the classification_report processing
    df.related.unique()
    df.replace(2,1,inplace=True)
    
    X = df["message"].values
    Y = df.drop(columns=["id","message","original","genre"],axis=1).values
    category_names=list(df.columns[4:])
    return X,Y,category_names


def tokenize(text):
    clean_tokens=[]
    lemmatizer = WordNetLemmatizer()
    lower_text=text.lower()
    p=re.compile("["+re.escape(string.punctuation)+"]")
    normalized_text=p.sub("",lower_text)
    word_tokens=word_tokenize(normalized_text)
    # Remove stop words
    stop_words = stopwords.words('english')
    filtered_words = [w for w in word_tokens if not w in stop_words]
    for tok in filtered_words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
    
def build_model():
    pipeline_simple = Pipeline([
        ("vect",CountVectorizer(tokenizer = tokenize)),
        ("tfidf",TfidfTransformer()),
        ("clf",MultiOutputClassifier(RandomForestClassifier()))
        #("clf",MultiOutputClassifier(KNeighborsClassifier()))
    ])
    parameters = {
        #'clf__estimator__n_estimators': [20,50,80],
        'clf__estimator__min_samples_leaf': [5, 10, 20],
        'clf__estimator__max_features': [0.5,1,"log2"],
        #'vect__max_df':[1, 5,10]
    }

    cv = GridSearchCV(pipeline_simple, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_test_pred = model.predict(X_test)
    report_data=classification_report(Y_test,y_test_pred,target_names=category_names)
    print("classification_report:",report_data)


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model, model_file)


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
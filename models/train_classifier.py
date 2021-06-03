import sys
import pandas as pd
import numpy as np
import re
import nltk
from sqlalchemy import create_engine
nltk.download(['punkt', 'wordnet'])
from sklearn.ensemble import GradientBoostingClassifier
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score,\
precision_score, recall_score, make_scorer
import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse',engine)
    X = df.message.values
    Y = df.drop(columns=['id', 'message', 'original', 'genre']).values
    category_names = np.array(df.drop(columns=['id', 'message', 'original', 'genre']).columns)
    return X,Y,category_names
    pass


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    pass
    pass


def build_model():
    pipeline =Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator = RandomForestClassifier(n_jobs=-1)))
    ])
    pipeline.get_params()
    parameters = {
        'clf__estimator__min_samples_split':[10,15],
        'clf__estimator__max_depth':[None,10,20],
        'clf__estimator__n_estimators': [10,20]
        }

    cv = GridSearchCV(pipeline, param_grid=parameters,cv=3,verbose=3) 
    return cv
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i,col in enumerate(category_names):
        print('------------------------')
        print(col)
        print(classification_report(Y_test[i], y_pred[i]))
        print("------------------------------------------------------\n")
     
      
    
    pass


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


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
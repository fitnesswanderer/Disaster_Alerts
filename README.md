# Disaster Response Pipeline Project

## Table of contents
Project Motivation

File structure

Installation

Licensing and Acknowledgements

## Project Motivation
Disasters are vulnerable situation and respondents needs help  immediately.The platform helps essential services and first responders to quickly identify relevant messages at the time of crisis to assess the situation and deploy adequate forces where needed.
## File structure
- app
 
| - template

|- master.html  # main page of web app

|- go.html  # classification result page of web app

|- run.py  # Flask file that runs app

- data

|- disaster_categories.csv  # data to process

|- disaster_messages.csv  # data to process

|- process_data.py

|- InsertDatabaseName.db   # database to save clean data to

- models

|- train_classifier.py

|- classifier.pkl  # saved model 

- README.md

In short: 
Build an ETL (Extract, Transform, Load) Pipeline to repair the data.

Build a supervised learning model using a machine learning Pipeline.

Build a web app that does the following:

Takes an input message and gets the classification results of the input in several categories.
Displays visualisations of the training datasets.

## Installation
python (>=3.6)

pandas

numpy

sqlalchemy

sys

plotly

sklearn

joblib

flask

nltk

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
       `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing and Acknowlegements
Thank you Figure Eight for providing the dataset. Udacity for giving the structure of code to help in implementing ETLand ML pipeline and deployment using flask.I would like to thank Maria Vaghani and Evans Doe for their help. All files in this repository are free to use.



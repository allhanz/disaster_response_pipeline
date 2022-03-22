# Disaster Response Pipeline Project

### project overview
 This project is part of the Udacity Data Scientist Nanodegree Program: Disaster Response Pipeline Project and the goal was to apply the data engineering skills learned in the course to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

### the detaset discription
 the dataset which is used in this project is provided by Figure Eight and also availale in this [site] (https://appen.com/datasets-resource-center/). The dataset contains over 20,000 messages not only including some disaster information such as earthquake, floods, super-storm and so on. but also includes news articles. 
 here is the screenshort for the message and categories dataset.
 1. message dataset:
  ![alt text](/readme_images/message_dataset.png "mess-labels")
 2. categories dataset:
  ![alt text](/readme_images/categories_dataset.png "mess-labels")

### Project Components
There are three components in this project as follows:
#### 1) ETL Pipeline
In a Python script, process_data.py, including the codes for: 
 1) Loads the messages and categories datasets
 2) Merges the two datasets
 3) Cleans the data
 4) Stores it in a SQLite database
#### 2) ML Pipeline
In a Python script, train_classifier.py, including the codes for:

 1) Loads data from the SQLite database
 2) Splits the dataset into training and test sets
 3) Builds a text processing and machine learning pipeline
 4) Trains and tunes a model using GridSearchCV
 5) Outputs results on the test set
 6) Exports the final model as a pickle file
#### 3) Flask Web App
In a Python script, run.py, including the codes for:
 1) build a web app for data visualization by mainly using plotly

### file explanation for this repo
├── ETL Pipeline Preparation.ipynb # ipynb file includes the code for how to pre-process the dataset \
├── ML Pipeline Preparation.ipynb # ipynb file incldues the code for how to build, train, and estimate the model by using the pre-processed dataset which had been storaged in the database \
├── README.md # includes the detailed information about this project, such as what kinds of packages will be used, how to run the web server tec \
├── app \
│   ├── run.py # Flask file that runs app \
│   └── templates \
│       ├── go.html # classification result page of web app \
│       └── master.html # main page of web app \
├── data \
│   ├── DisasterResponse.db # stored the pre-processed dataset \
│   ├── disaster_categories.csv # the dataset csv file which is used in this project \
│   ├── disaster_messages.csv # the dataset csv file which is used in this project \
│   └── process_data.py # process the dataset and stored in the database \
├── models \
│   ├── classifier.pkl # the best model weight files saved \
│   └── train_classifier.py #  train the mode and save the best model into pkl file \
└── readme_images  \
    ├── example_usage.png # give a example for how to use this disaster pipeline in this webpage  \
    ├── main_web_screen.png # the screenshort when the web server runs successfully \
    ├── categories_dataset.png # screenshort for the categories dataset  \
    └── message_dataset.png # screenshort for the message dataset \

### packages used in this project
 pandas==0.23.3
 nltk==3.2.5
 sqlalchemy==1.2.19
 numpy==1.12.1
 scikit-learn==0.19.1
 flask==0.12.5
 plotly==2.0.15

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
 ![alt text](/readme_images/main_web_screen.png "mess-labels")

4. Input some texture message just you like. and you will get the similar results as follows
 ![alt text](/readme_images/example_usage.png "mess-labels")




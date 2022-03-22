# Disaster Response Pipeline Project

### project overview
  this project is to identify the relationship between the disaster and the actual texture messages by using NLP tech.  

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




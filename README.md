# Disaster Response Pipeline Project

### Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Project files](#project_files)
4. [Project details](#project_details)
5. [Results](#results)
6. [Local setup instructions](#local_setup)
7. [Licensing, Authors, and Acknowledgements](#licensing)


## Overview <a name="overview"></a>

Project to classify disaster response messages using python pipelines and machine learning to classify 36 response categories. This project is based on Udacity's Data Science Nanodegree program.

Process
1. Analyse the classified disaster response raw data
2. Create database with cleaned and processed raw data
3. Create model for classification and prediction (NLP, KNN classifier, RandomForest classifier)
4. Train model and optimize with GridSearchCV
5. Build webapp to use model for prediction of inputted words for disaster response

## Installation <a name="installation"></a>

1. Download all files 
2. python app/run.py to run the code

Note, refer to local setup instructure. 
Comment out "app.run(host='0.0.0.0', port=3001, debug=True)" from app/run.py to run online.

### Project files <a name="project_files"></a>
data folder:
- process_data.py: Main file to run in order to clean and store data in database (refer to local setup)
- disaster_categories.csv: Categories csv data to clean and process
- disaster_messages.csv: Messages csv data to clean and process
- DisasterResponse.db: Database file that has been generated from cleaning and processing
- ETL Pipeline Preparation.ipynb: Jupyter notebook file for testing and initial analysis. Detailed works and experimentation can be found here
- disaster_response_pete_proj2.db: Tested output database from initial playground and experimentation analysis

models folder:
- train_classifier.py: Main file to run in order to classify and create a model for prediction
- classifier-knn.pkl: Classifier model which has been created based on KNN classifier
- classifier.pkl: Initial tested classifier model created with RandomForestClassifier for experimentation
- ML Pipeline Preparation.ipynb: Jupyter notebook file for testing and initial analysis on ML pipeline preparation. Detailed works and experimentation can be found here
- ML Analysis playground.ipynb: Extended analysis file for testing the models used.
- model1.pkl: Test model 1 pickle file from experimentation
- model2.pkl: Test model 2 pickle file from experimentation

prep_notebooks folder:
- Test files for this project and experimentation analysis

proj_app folder:
- Project application files to run with the init file and run app file

proj_methods folder:
- custom methods used for the classification (has to make separate module in order to work with pickle and unpickle file appropriately)

### Project details <a name="project_details"></a>
Project required multiple analysis throughout.
- Cleaning and processing of data
- Natural language processing to translate words to numerical values for classification and prediction
- Creating/saving/loading data into local databases for reference and usage
- Modelling and optimization to achieve the best results
- Pipeline creationg and optimisation with GridSearchCV
- Saving model into pickle files for reloading and usage in web applications
- Creating and running web applications with visualised plots on the outputting predicted model and results

Learnings
- Fitting models takes a long time so take time to understand the different classifiers and selecting appropriate parameters for optimization
- Natural language processing can comes in multiple forms. Having more information breakdown from words can be useful but be careful of the context that we are looking to extract data to use to ensure no overfitting
- Creation of multiple transforms is useful in pipelines to aggregate and run different classifications together for analysis
- Ensure that the results are valid and test outputting them first (tested with a classifier and it outputted one result since the classification accuracy is high, be careful and actually check the raw results first)
- Pickle file usage to import module (note in the app file, to change location when calling the database). Resolved with creating module structure for the proj_methods used relating to tokenize and the custom transformers.
- Antoher issue is with too large file. Classifier with KNN has large file so resolving with git lfs
- Pickle file issues with setting up for Heroku upload and deploy as running locally works but gunicorn and deploy fails

## Results<a name="results"></a>
Output results can be found at proj-disaster-resp-test1.herokuapp.com
For more information, check the Jupyter notebook files or contact the project's author.

### Local setup instructions <a name="local_setup"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the root directory to run your web app.
    `python proj_app.py`

3. Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credit to Figure Eight data and Udacity Data Science Nanodegree program.
https://www.figure-eight.com/ 
https://www.udacity.com/course/data-scientist-nanodegree--nd025







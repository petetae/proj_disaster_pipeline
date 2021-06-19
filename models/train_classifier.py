import sys
import pandas as pd
import numpy as np
import sqlalchemy as db

# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('words')
nltk.download('vader_lexicon')

# import statements
import pandas as pd
from nltk import word_tokenize, sent_tokenize
# , pos_tag, ne_chunk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import joblib

def load_data(database_filepath):
    """ Load data from database file path and then return the appropriate information in separate columns
    Input:
    - database_filepath - Filepath of the database to load

    Output:
    - X - DataFrame of the message column
    - Y - DataFrame without message, id, original, and genre column
    - column_names - column names of the different attributes for prediction
    """
    # Create engine with sqlite with input database file name
    engine = db.create_engine('sqlite:///{}'.format(database_filepath))
    
    # Read the data sql table to DataFrame
    df = pd.read_sql_table(database_filepath,engine)

    # Separate the columns appropriately for returning
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    
    return X, Y, Y.columns

def tokenize(text):
    """ Tokenize text for count vectorizer
    Method to normalize, lemmatize, and tokenize text through breaking down words

    Input:
    - Text string

    Output:
    - Tokenized text list with lemmatize, normalize and tokenize words
    """
    # Retuning the lemmatized, normalized, and tokenized text
    return [WordNetLemmatizer().lemmatize(w).strip() for w in word_tokenize(text.lower())]
    

class HelpWordExtractor(BaseEstimator, TransformerMixin):
    """ Transformer to extract out the word "help" and returns 1 or 0 to support help response classifications
    Input:
    - Sentence list for analysis and prediction

    Output:
    - DataFrame Series of 1 or 0s nothing whether there is the word "help" in message or not
    """
    # To identify specific words for help and label them appropriately
    def contains_help(self, text):
        # Breakdown to each sentence for tokenization
        sentence_list = nltk.sent_tokenize(text)

        # Loop through the list to extract each string
        for sentence in sentence_list:
            # Return 1 if there is word "help" and 0 if not
            if 'help' in sentence:
                return 1
            else:
                return 0
        # Return 0 if sentence is empty or so
        return 0

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # Apply the contains_help function to the inputted data and transform it as noted
        X_tagged = pd.Series(X).apply(self.contains_help)
        return pd.DataFrame(X_tagged)

    
class WordLengthExtractor(BaseEstimator, TransformerMixin):
    """ Transformer to output the list of word length of the message 
    This is to further add information regarding the inputted message whether there is a relationship between wordlength and the model

    Input:
    - Sentence list for analysis and prediction

    Output:
    - DataFrame Series of message length
    """
    # To see word length whether longer message have more distress in them or not
    def word_length(self, text):
        return len(text)

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # Apply the word_length function to the inputted data and transform it as noted
        X_tagged = pd.Series(X).apply(self.word_length)
        return pd.DataFrame(X_tagged)
    
class SentimentSentenceExtractor(BaseEstimator, TransformerMixin):
    """ Transformer to output the compound sentiment score of the sentence
    Sentiment is a measure of whether the sentence has a positive or negative words in it.
    This is done through using the SentimentIntensityAnalyzer from nltk.sentiment

    Input:
    - Sentence list for analysis and prediction

    Output:
    - DataFrame Series of compound scores
    """    
    # To see word length whether longer message have more distress in them or not
    def sentiment_analyser(self, text):
        return SentimentIntensityAnalyzer().polarity_scores(text)['compound']

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # Apply the sentiment_analyser function to the inputted data and transform it as noted
        X_tagged = pd.Series(X).apply(self.sentiment_analyser)
        return pd.DataFrame(X_tagged)

def build_model():
    """ Build model and pipeline to be used for evaluate model and prediction
    Output:
    - Model that has been created with pipeline to run wiht Gridsearchcv for best parameters



    TODO -> Need to use KNN classifier instead
    Note, for improvements refer to the Jupyter notebook for testing and optimizing models for usage, attached to this project
    Optimising model in jupyter notebook. Here, inputting the most optimised parameters (list out the test we tried and show results)
    The most optimal tested, balancing with speed to model is the RandomForestClassifier

    Tested with RandomForestClassifier, KNeighborsClassifier, DecisionTreeClassifier, SVC and chose RandomForestClassifier based on results
    Transformers used and tested can see analysis in the attached Jupyter notebook
    """
    # Create pipeline to builde with GridSearchCV with various transformers
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('hwe', HelpWordExtractor()),
            ('wle', WordLengthExtractor()),
            ('sse', SentimentSentenceExtractor()),
        ])),
        # ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ], verbose=True)

    # Setup the parameters that we want to test run with the pipeline through GridSearchCV
    parameters = {
        # 'clf__estimator__max_depth' : [2,5],
        # 'clf__estimator__n_estimators' : [10,20],
        # 'clf__estimator__max_features' : [1]

        # Tested with RandomForest and KNN. Choose to go with KNN with these parameters optimised
        'clf__estimator__n_neighbors': [5,10],
        'clf__estimator__leaf_size': [30, 50]

        # For testing
        # 'clf__estimator__n_neighbors': [5],
        # 'clf__estimator__leaf_size': [50]
    }

    # Run the pipeline with GridSearchCV to find the most optimal parameters and return this model
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=5)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluate model for the results and accuracy of it
    Input:
    - model - Model which has been fitted
    - X_test - X test data that is to be tested and evaluated
    - Y_test - Y test data that is to be tested and evaluated
    - category_names to note the different category names 

    Output:
    - Printed output of the classification_report for the results
    - Printed output of the best parameters from the model
    - Printed output of the average accuracy score from the test and prediction results on all categories
    - Printed output of the best estimator which was used to run the test
    - Printed output of the best score from the model generated

    For further inforamtion, please visit the attached Jupyter Notebook for the analysis and usage
    """
    
    # Predict output with test data
    y_pred = model.predict(X_test)

    # Create dataframe of results for ease of access and handling
    y_pred_df = pd.DataFrame(y_pred, columns=category_names)

    # Classification report analysis for accuracy checking for all data in the category_names, and print out classification_report
    for col_name in category_names:
        print(classification_report(Y_test[col_name],y_pred_df[col_name]))

    # Print out other results and features
    print("Best params ----- = ", model.best_params_)
    print("Accuracy (average): ", (Y_test == y_pred).mean().mean())
    print("Best estimator ------ ", model.best_estimator_)
    print("Best score ------ ", model.best_score_)

def save_model(model, model_filepath):
    """ Save the model to a pickle file for usage later on
    Input:
    - model - The resulting model that has been created and generated
    - model_filepath - The file path to the model that is going to be used for testing (example model_filepath = "model1.pkl" )

    Output:
    - Returns True or False based on success of the saving
    - Prints success/error message
    """
    # Save model as pickle file
    # Try and catch the error to see whether it will fail to open or create or not
    try:
        # Creates and saves the inputted model as a pickle file
        # with open(model_filepath, 'wb') as file:  
        #     pickle.dump(model, file)
        joblib.dump(model, model_filepath)
        print("Saves file {} successfully as a model pickle file".format(model_filepath))

    except Exception as e:
        # Catches the exception and then prints out if there is an error
        print("Error in saving the file. Please resolve the error and try again")
        print(e)

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
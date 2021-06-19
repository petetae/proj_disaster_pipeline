import sys
import pandas as pd
import numpy as np
import sqlalchemy as db

def load_data(messages_filepath, categories_filepath):
    """ Reads in messages and catogies data filepath 
        Merges data together to return Dataframe
    
    Input:
    - messages_filepath
    - categories_filepath

    Output:
    - DataFrame with messages and categories information
    
    Function will try to read the data in and if fail, will print fail response and return []
    """
    # Try read csv data files. If succeed, then attempt to merge on 'id' field and if fail, return []
    try:
        messages = pd.read_csv(messages_filepath)
        categories = pd.read_csv(categories_filepath)
        return messages.merge(categories,on='id',how='inner')
    except Exception as e:
        print("ERROR LOADING FILE")
        print("RESOLVE ERROR AND TRY AGAIN!")
        print(e)
        return []

def clean_data(df):
    """ Cleans the data from the DataFrame respectively
    Function to clean the data with the following process:
    1. Clean by splitting ;
    2. Clean by renaming columns to be different categories
    3. Clean by converting category values to 0 or 1 number
    4. Clean by removing duplicate data
    5. Clean by removing unknown and unrelated data

    Input:
    - DataFrame of messages and categories format

    Output:
    - DataFrame of cleaned data

    Note, in case error dataframe input, it will return []
    """
    # Check if dataframe input is empty, then return empty
    if df is []:
        print("EMPTY DATAFRAME INPUTTED!!!")
        return [] 
    print(".....Attempting to clean data.....")
    print(".....Splitting dataframe of categories column by ;.....")
    # Split the DataFrame into different 36 individual categories
    categories = df.categories.str.split(';',expand=True)
    
    print(".....Renaming category column names appropriately.....")
    # Renaming the category columns by first selecting row of categories column for renaming
    row = categories.iloc[0]
    # Extract the column name by removing last 2 character of the string
    category_colnames = row.apply(lambda a : a[:-2])
    # Rename the columns of `categories`
    categories.columns = category_colnames
    
    print(".....Convert category values to just numbers 0 or 1.....")
    # Extract data from categories DataFrame
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda a : int(a[-1]))
    # Drop current categories column in df Dataframe 
    if 'categories' in df.columns:
        df.drop(['categories'],axis=1,inplace=True)
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    print(".....Clean duplicate data.....")
    print("Current duplicate count = {}".format(df.duplicated().sum()))
    print(".....Dropping duplicate data.....")
    # Drop duplicates rows
    df.drop_duplicates(inplace=True)
    print("Current duplicate count = {}".format(df.duplicated().sum()))

    print(".....Remove unrelated data (data should only contain 0 or 1 but found '2' in related column).....")
    # Remove unrelated data as cannot categories. this is becase
    # 1. if it gets categoried as 2 for related, might mean something different
    # 2. as it can mean something different, better to remove than to mix and cause outlier to classify
    # 3. should validate with the data on what this mean whether it is jsut mislabeled or a new category group etc
    print("Check 'related' column data type {}".format(df.related.unique()))
    print(".....Removed '2' outlier from DataFrame")
    # Remove data which is '2' in 'related' column
    df = df[~(df['related']== 2)]
    print("Check 'related' column data type {}".format(df.related.unique()))
    return df

def save_data(df, database_filename):
    """ Saves data from DataFrame to database of the inputted filename
    Takes in the DataFrame and attempts to save in the database. Check return true or false for success.

    Input:
    - DataFrame of the cleaned messages and categories data
    - Database file name that is going to be saved to

    Output:
    - True or False for success (optional)
    """

    print(".....Create sqlite engine with database filename.....")
    # Create engine to save data to sqlite database with the inputted name
    engine = db.create_engine('sqlite:///{}'.format(database_filename))
    # Try to save the database. Returns True if success and False if fails
    try:
        print(".....Attempt to create and save database.....")
        # Attempts to save database to the system
        df.to_sql(database_filename, engine, index=False)
        print(".....Database write success.....")
        return True
    except Exception as e:
        print("Something wrong with trying to save database!")
        print("Resolve error and try again!")
        print(e)
        return False

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
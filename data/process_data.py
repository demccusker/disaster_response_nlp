import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads two files of messages and categories into a single concatenated dataframe
    inputs: messages_filepath, categories_filepath
    outputs: dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,how="left",on="id")
    return df

def clean_data(df):
    '''
    Cleans inputed dataframe by seperating out columns for categories into ML readable format and dropping duplicates
    input: df
    output: cleaned df
    '''
    categories = df['categories'].str.split(";",expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str.strip().str[-1]
        categories[column] = categories[column].astype(int)
    categories = categories[categories['related'] != 2]
    df.drop(labels='categories',axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filepath):
    '''
    Saves data to a SQL database
    inputs: dataframe and database_filename
    returns: nothing 
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql(database_filepath, engine, index=False,if_exists='replace') 


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
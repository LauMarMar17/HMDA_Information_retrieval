"""
This script...
"""
import pandas as pd 
from functions import *

def get_data():
    train_df = pd.read_csv('../data/Movies/train_data.csv', header=None)
    test_df = pd.read_csv('../data/Movies/test_data.csv', header=None)
    # Eliminar primera columna
    train_df = train_df.drop(columns=[0])
    test_df = test_df.drop(columns=[0])
    # Asignar nombres a las columnas: 'title', 'genre', 'plot'
    train_df.columns = ['title', 'genre', 'plot']
    test_df.columns = ['title', 'plot']
    return train_df, test_df

def main():
    # get dataframes (data)
    train_df, test_df = get_data()
    # process text
    train_df['plot'] = train_df['plot'].apply(process_text)
    pass
    


if __name__ == "__main__":
    main()
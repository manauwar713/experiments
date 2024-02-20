import pandas as pd
import numpy as np
import pathlib
import os
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    df = pd.read_csv(path)
    return df
def encoding(df):
    label_encoder = LabelEncoder()
    columns_to_encode = ['workclass','marital-status','occupation','relationship','race','gender','native-country','income']
    for coloumn in columns_to_encode:
        df[coloumn] = label_encoder.fit_transform(df[coloumn])
    return df
def split_data(df,test_split,seed):
    
    train,test = train_test_split(df,test_size = test_split,random_state = seed)
    return train,test
def save_data(train,test,output_path):
    pathlib.Path(output_path).mkdir(parents=True,exist_ok=True)
    train.to_csv(output_path + '/train.csv',index = False)
    test.to_csv(output_path + '/test.csv',index = False)
    

    



        
        
def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + "/params.yaml"
    params = yaml.safe_load(open(params_file))["make_dataset"]
    
    
    data_path = home_dir.as_posix() + '/data/raw/cleaned_adult.csv'
    output_path = home_dir.as_posix() + '/data/processed'
    
    
    data = load_data(data_path)
    print(data.shape)
    data = encoding(data)
    print(data.shape)
    train_data,test_data = split_data(data,params['test_split'],params['seed'])
    print(train_data.shape)
    save_data(train_data,test_data,output_path)
    
    
if __name__ == "__main__":
    main()
    
        

    

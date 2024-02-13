import pandas as pd
import numpy as np
import pathlib
import os
import yaml


def clients_data(path,num_clients,output_path,seed):
    df = pd.read_csv(path)
    gender_counts = df['gender'].value_counts()
    total_samples = gender_counts.sum()
    male_proportion = gender_counts['Male'] / total_samples
    female_proportion = gender_counts['Female'] / total_samples
    
    concentration_parameters = [male_proportion, female_proportion]
    num_clients = num_clients
    
    dirichlet_samples = np.random.dirichlet(concentration_parameters, size=num_clients)
    
    for i, proportions in enumerate(dirichlet_samples):
    
        male_samples = int(proportions[0] * total_samples)
        female_samples = int(proportions[1] * total_samples)
        
        
        male_data = df[df['gender'] == 'Male'].sample(male_samples, replace=True)
        female_data = df[df['gender'] == 'Female'].sample(female_samples, replace=True)
        
        
        client_data = pd.concat([male_data, female_data])
        
        
        
        
        pathlib.Path(output_path).mkdir(parents=True,exist_ok=True)
        output_dir = os.path.join(output_path,f'client_{i+1}.csv')
        
        client_data.to_csv(output_dir,index=False)
        
        
def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + "/params.yaml"
    params = yaml.safe_load(open(params_file))["make_dataset"]
    
    
    data_path = home_dir.as_posix() + '/data/raw/cleaned_adult.csv'
    output_path = home_dir.as_posix() + '/data/processed'
    
    clients_data(data_path,params['num_clients'],output_path,params['seed'])
    
if __name__ == "__main__":
    main()
    
        

    

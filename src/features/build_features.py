import pandas as pd
import numpy as np
import pathlib
import os
import yaml


def clients_data(path,num_clients,output_path,seed):
    df = pd.read_csv(path)
    
    gender_counts = df['gender'].value_counts()
    total_samples = gender_counts.sum()
    male_proportion = gender_counts[1] / total_samples
    female_proportion = gender_counts[0] / total_samples
    
    concentration_parameters = [male_proportion, female_proportion]
    num_clients = num_clients
    df_temp = df.copy()
    
    dirichlet_samples = np.random.dirichlet(concentration_parameters, size=num_clients)
    print(dirichlet_samples)
    
    for i, proportions in enumerate(dirichlet_samples):
        samples = total_samples/ num_clients
    
        male_samples = int(proportions[0] * samples)
        female_samples = int(proportions[1] * samples)
        print(male_samples)
        
        if i < num_clients-1:
            male_data = df_temp[df_temp['gender'] == 1].sample(male_samples, replace=False)
            female_data = df_temp[df_temp['gender'] == 0].sample(female_samples, replace=False)
            df_temp.drop(index=male_data.index,inplace = True)
            df_temp.drop(index=female_data.index,inplace=True)
        else:
            male_data = df_temp[df_temp['gender'] == 1]
            female_data = df_temp[df_temp['gender'] == 0]
            
        
        
        client_data = pd.concat([male_data, female_data])
        
        
        
        
        pathlib.Path(output_path).mkdir(parents=True,exist_ok=True)
        output_dir = os.path.join(output_path,f'client_{i+1}.csv')
        
        client_data.to_csv(output_dir,index=False)
        
def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + "/params.yaml"
    params = yaml.safe_load(open(params_file))["make_dataset"]
    
    
    data_path = home_dir.as_posix() + '/data/processed/train.csv'
    output_path = home_dir.as_posix() + '/data/interim'
    
    clients_data(data_path,params['num_clients'],output_path,params['seed'])
    
if __name__ == "__main__":
    main()
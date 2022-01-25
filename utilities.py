import os
import pickle
import joblib
from joblib import dump

def data_saver(config,save_list,XorY):

    saveFlag = 'y'

    #count existing files
    count = 1
    df_dir = "data/USER/DataFrames"
    
    filename = 'df_'+ XorY + str(count)
    try:
        for path in os.listdir(df_dir):
            print(path)
            while True:
                if str(count) in path:
                    count += 1
                    filename = 'df_'+ XorY + str(count)
                    print(filename)
                else:
                    break 
    except:
           os.makedirs(os.path.join('./',df_dir))
        # if os.path.isfile(os.path.join(df_dir, path)) and XorY in os.path:
        #     count += 1

    for fname in os.listdir(df_dir):
        print(os.path.join(df_dir, fname))
        if XorY in fname:
            with open(os.path.join(df_dir, fname), 'rb') as f:
                print(os.path.join(df_dir, fname))
                pickle_list = pickle.load(f)

                if pickle_list[0] == config:
                    print(config)
                    saveFlag = 'n'
                    while True:
                            saveFlag = input("Dataframe with current configuration already exists:  " + fname + "| Do you want to overwrite? (y/n) ")
                            if saveFlag not in ["y","n"]:
                                print("Sorry, please enter y/n:")
                                continue
                            else:
                                break

                    if saveFlag == 'y':
                        os.remove(os.path.join(df_dir,fname))
                        filename = fname

    if saveFlag == 'y':
        with open(os.path.join(df_dir, filename),'wb') as f:        
            pickle.dump(save_list,f)

def data_loader(load_config,XorY):
    df_directory = "data/USER/DataFrames"
    config_match = False

    for filename in os.listdir(df_directory):
        with open(os.path.join(df_directory, filename), 'rb') as f:
            pickle_list = pickle.load(f)

            if pickle_list[0] == load_config:
                if XorY in filename:
                    print("Data loaded from ",os.path.join(df_directory, filename))
                    all_stations = pickle_list[1]
                    ind_stations = []
                
                    for station in pickle_list[2:]:
                        ind_stations.append(station)
                        config_match = True

    if config_match == False:
        print(load_config)
        raise NameError('Configuration does not match any current dataframes. Check configuration or generate new data.')

    return all_stations,ind_stations

def model_saver(model,model_name,name):
    df_dir = "data/USER/Models/"
    
    if 'sklearn' in model_name:
        dir_path = os.path.join(df_dir, model_name)  # will return 'feed/address'
        try:
            os.makedirs(dir_path)
        except:
            path_exists = True
        with open(os.path.join(dir_path,name),'wb') as f:        
            dump(model,f)

         
def model_loader(model,model_name,name):
    model_type = type(model)
    df_dir = "data/USER/Models/"

    if 'sklearn' in model_name:
        dir_path = os.path.join(df_dir, model_name,name)
        with open(dir_path,'rb') as f:
            model = joblib.load(f)
    
    return model
# Average of best results
#%%
import os
import pandas as pd

# Load Random Forest Data

filepath = "/Users/jonathanerskine/Courses/Machine Learning Paradigms/CWK/MB_2021/data/USER/Submissions/RF_groupD_all_stationssubmission.csv"
with open(filepath, 'r') as f:
       data_RF = pd.read_csv(f)

filepath = "/Users/jonathanerskine/Courses/Machine Learning Paradigms/CWK/MB_2021/data/USER/Submissions/linearfullsubmission.csv"
with open(filepath, 'r') as f:
       data_LM = pd.read_csv(f)
# %%
RF_bikes = data_RF['bikes'].values
LM_bikes = data_LM['bikes'].values

res_bikes = [round((x+y)/2) for x,y in zip(RF_bikes,LM_bikes)]

print(RF_bikes[:5],LM_bikes[:5],res_bikes[:5])
# %%


mean_test = pd.DataFrame({"bikes":res_bikes})
mean_test.index.name = 'Id'
mean_test.round()
mean_test.index+=1
mean_test.to_csv('data/USER/Submissions/linear' + 'mean_model' +'submission.csv',header=["bikes"]) #,quoting=csv.QUOTE_NONE)


# %%

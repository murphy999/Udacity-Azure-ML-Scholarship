# System Libraries
import os

# Data Wrangling Libraries
import numpy as np
import pandas as pd
import argparse

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib

# Azure Libraries
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
#from azureml.core import Workspace, Dataset

#subscription_id = ''
#resource_group = ''
#workspace_name = ''
#workspace = Workspace(subscription_id, resource_group, workspace_name)

#dataset = Dataset.get_by_name(workspace, name='mpg')
#data = dataset.to_pandas_dataframe()

data_link = ""
ds = TabularDatasetFactory.from_delimited_files(path=data_link)

def clean_data(data):
    
    # Clean and one hot encode data
    df = data.to_pandas_dataframe().dropna()
    df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')

    df['Partner'] = pd.Series(np.where(df.Partner.values == 'Yes',1,0))
    df['Dependents'] = pd.Series(np.where(df.Dependents.values == 'Yes',1,0))
    df['PaperlessBilling'] = pd.Series(np.where(df.PaperlessBilling.values == 'Yes',1,0))
    df['PhoneService'] = pd.Series(np.where(df.PhoneService.values == 'Yes',1,0))
    df['MultipleLines'] = pd.Series(np.where(df.MultipleLines.values == 'Yes',1,0))
    df['OnlineSecurity'] = pd.Series(np.where(df.OnlineSecurity.values == 'Yes',1,0))
    df['OnlineBackup'] = pd.Series(np.where(df.OnlineBackup.values == 'Yes',1,0))
    df['DeviceProtection'] = pd.Series(np.where(df.DeviceProtection.values == 'Yes',1,0))
    df['TechSupport'] = pd.Series(np.where(df.TechSupport.values == 'Yes',1,0))
    df['StreamingMovies'] = pd.Series(np.where(df.StreamingMovies.values == 'Yes',1,0))
    df['StreamingTV'] = pd.Series(np.where(df.StreamingTV.values == 'Yes',1,0))
    
    df = pd.get_dummies(df,columns=['gender'],drop_first= True)
    df = pd.get_dummies(df,columns=['InternetService'],drop_first= True)
    df = pd.get_dummies(df,columns=['PaymentMethod'],drop_first= False)
    df = pd.get_dummies(df,columns=['Contract'],drop_first= True)

    y_df = df.pop("Churn").apply(lambda s: 1 if s == "Yes" else 0)
    
    return x_df,y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="The number of trees in the forest")
    parser.add_argument('--max_depth', type=int, default=5, help="The maximum depth of the tree")
    parser.add_argument('--min_samples_split', type=int, default=3, help="The minimum number of samples required to split an internal node")

    args = parser.parse_args()

    x, y = clean_data(ds)

    # TODO: Split data into train and test sets.
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=53)

    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, min_samples_split=args.min_samples_split).fit(x_train, y_train)
    
    y_prob = model.predict_proba(x_test)
    y_pred = y_prob[:, 1]

    weighted_auc = roc_auc_score(y_test,y_pred,average='weighted')
    run.log("AUC_weighted", np.float(weighted_auc))
    
    joblib.dump(model,'./model.joblib')

if __name__ == '__main__':
    main()
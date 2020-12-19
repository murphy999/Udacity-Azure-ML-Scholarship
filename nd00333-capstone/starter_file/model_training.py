# System Libraries
import os

# Data Wrangling Libraries
import numpy as np
import pandas as pd
import argparse

# Machine Learning Libraries
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Azure Libraries
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

data_link = "https://raw.githubusercontent.com/murphy999/Udacity-Azure-ML-Scholarship/master/nd00333-capstone/starter_file/Telco-Customer-Churn.csv"
ds = TabularDatasetFactory.from_delimited_files(path=data_link)


# +
#from azureml.core import Workspace, Dataset
#subscription_id = 'de8aba62-c352-42be-b980-2faedf08ead8'
#resource_group = 'aml-quickstarts-130769'
#workspace_name = 'quick-starts-ws-130769'
#wrkspace = Workspace(subscription_id,resource_group,workspace_name)

#temp = Dataset.get_by_name(wrkspace, name='customer')
#datatset = temp.to_pandas_dataframe()
# -

def clean_data(data):
    
    # Clean and one hot encode data
    df = data.to_pandas_dataframe().dropna()
    #df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')

    df['Partner'] = df.Partner.apply(lambda s: 1 if s == True else 0)
    df['Dependents'] = df.Dependents.apply(lambda s: 1 if s == True else 0)
    df['PaperlessBilling'] = df.PaperlessBilling.apply(lambda s: 1 if s == True else 0)
    df['PhoneService'] = df.PhoneService.apply(lambda s: 1 if s == True else 0)
    df['MultipleLines'] = df.MultipleLines.apply(lambda s: 1 if s == "Yes" else 0)
    df['OnlineSecurity'] = df.OnlineSecurity.apply(lambda s: 1 if s == "Yes" else 0)
    df['OnlineBackup'] = df.OnlineBackup.apply(lambda s: 1 if s == "Yes" else 0)
    df['DeviceProtection'] = df.DeviceProtection.apply(lambda s: 1 if s == "Yes" else 0)
    df['TechSupport'] = df.TechSupport.apply(lambda s: 1 if s == "Yes" else 0)
    df['StreamingMovies'] = df.StreamingMovies.apply(lambda s: 1 if s == "Yes" else 0)
    df['StreamingTV'] = df.StreamingTV.apply(lambda s: 1 if s == "Yes" else 0)
    
    df = pd.get_dummies(df,columns=['gender'],drop_first= True)
    df = pd.get_dummies(df,columns=['InternetService'],drop_first= True)
    df = pd.get_dummies(df,columns=['PaymentMethod'],drop_first= False)
    df = pd.get_dummies(df,columns=['Contract'],drop_first= True)
    
    y_df = df.pop("Churn").apply(lambda s: 1 if s == True else 0)
    
    return df,y_df

run = Run.get_context()

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

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", float(accuracy))
    
    joblib.dump(model,'outputs/model.joblib')

if __name__ == '__main__':
    main()

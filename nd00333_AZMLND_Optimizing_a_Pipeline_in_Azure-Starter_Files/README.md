# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
<p>In this project we have used UCI Bank Marketing dataset, which is related with direct marketing campaigns of a Portuguese baking institution. The classification goal is predict if the client will subscribe a term deposit (variable 'y'). <a href="https://archive.ics.uci.edu/ml/datasets/Bank+Marketing"> Read More </a></p>
<img src='https://github.com/murphy999/Udacity-Azure-ML-Scholarship/blob/master/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/images/pipeline.PNG'>
<p>In this project, we have used scikit-learn Logistic Regression and tuned the hyperparameters(optimal) using HyperDrive. We also used AutoML to build and optimize a model on the same dataset, so that we can compare the results of the two methods.
The best performing model was obtained through AutoML - <strong> VotingEnsemble </strong> with accuracy of <b>0.9177</b></p>

## Scikit-learn Pipeline
<ol>
  <li>Setup Training Script
    <ul>
      <li> Import data using <i>TabularDatasetFactory</i> </li>
      <li> Cleaning of data -  handling NULL values, one-hot encoding of categorical features and preprocessing of date </li>
      <li> Splitting of data into train and test data </li>
      <li> Using scikit-learn logistic regression model for classification </li>
    </ul>
  </li><br>
  <li>Create SKLearn Estimator for training the model selected (logistic regression) by passing the training script and later the estimator is passed to the hyperdrive                 configuration</li><br>
  <li> Configuration of Hyperdrive
    <ul>
      <li> Selection of parameter sampler </li>
      <li> Selection of primary metric </li>
      <li> Selection of early termination policy </li>
      <li> Selection of estimator (SKLearn) </li>
      <li> Allocation of resources </li>
      <li> Other configuration details </li>
    </ul>
  </li><br>  
  <li>Save the trained optimized model</li>
</ol>
<img src = 'https://github.com/murphy999/Udacity-Azure-ML-Scholarship/blob/master/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/images/hyperdrive_runs.PNG'>

<strong>Parameter Sampler</strong>
<p>The parameter sampler I chose was <i>RandomParameterSampling</i> because it supports both discrete and continuous hyperparameters. It supports early termination of low-performance runs and supports early stopping policies. In random sampling , the hyperparameter (C : smaller values specify stronger regularization, max_iter : maximum number of iterations taken for the solvers to converge) values are randomly selected from the defined search space. </p>

<strong>Early Stopping Policy</strong>
<p> The early stopping policy I chose was <i>BanditPolicy</i> because it is based on slack factor and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor compared to the best performing run. <a href = 'https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py&preserve-view=true#&preserve-view=truedefinition'>Read More</a></p>

## AutoML
<ol>
  <li> Import data using <i>TabularDatasetFactory</i></li>
  <li> Cleaning of data -  handling NULL values, one-hot encoding of categorical features and preprocessing of date </li>
  <li> Splitting of data into train and test data </li>
  <li> Configuration of AutoML </li>
  <li> Save the best model generated </li>
</ol>
<img src= 'https://github.com/murphy999/Udacity-Azure-ML-Scholarship/blob/master/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/images/automl_models.PNG'>
<p> The below snapshots gives the explanation of the best model prediction by highlighting feature importance values and discovering patterns in data at training time. It also shows differnt metrics and their value for model interpretability and explanation. </p>
<img src= 'https://github.com/murphy999/Udacity-Azure-ML-Scholarship/blob/master/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/images/ve1.PNG'>
<img src= 'https://github.com/murphy999/Udacity-Azure-ML-Scholarship/blob/master/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/images/ve2.PNG'>
<img src= 'https://github.com/murphy999/Udacity-Azure-ML-Scholarship/blob/master/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/images/automl_metric.png'>

## Pipeline comparison
<p>Both the approaches - Logistics + Hyperdrive and AutoML follow similar data processing steps and the difference lies in their configuration details. In the first approach our ML model is fixed and we use hyperdrive tool to find optimal hyperparametets while in second approach different models are automatic generated with their own optimal hyperparameter values and the best model is selected. In the below image, we see that the hyperdrive approach took overall <b>11m 51s</b> and the best model had an accuracy of <b>~0.9146</b> and the automl approach took overall <b>28m 58s</b> and the best model had an acccuracy of <b>~0.9177</b>.
</p>
<img src = 'https://github.com/murphy999/Udacity-Azure-ML-Scholarship/blob/master/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/images/comparison.PNG'>
<p> It is quite evident that AutoML results in better accurate model but takes time to find out one while the Logistic + Hyperdrive takes lesser time to find out an optimal hyperparameter values for a fixed model. Since we have used the same dataset and preprocessed the data in the same fashion we see that both the approaches generate model whose accuracy is very close.
</p>

## Future work
<ul>
 <li>To check or measure the fairness of the models</li>
 <li>Leverage additional interactive visualizations to assess which groups of users might be negatively impacted by a model and compare multiple models in terms of their              fairness and performance</li>
</ul>

## Proof of cluster clean up
<img src= 'https://github.com/murphy999/Udacity-Azure-ML-Scholarship/blob/master/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/images/cluster_cleanup.PNG'>

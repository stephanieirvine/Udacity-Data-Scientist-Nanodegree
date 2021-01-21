# Sparkify: Predicting Churn for a Music Streaming Service 

## Installations
 - NumPy
 - Pandas
 - Seaborn
 - Matplotlib
 - PySpark SQL
 - PySpark ML 
 
No additional installations beyond the Anaconda distribution of Python and Jupyter notebooks.

## Project Motivation
For this project I was interested in predicting customer churn for a fictional music streaming company: Sparkify. 

The project involved:
 - Loading and cleaning a small subset (128MB) of a full dataset available (12GB) 
 - Conducting Exploratory Data Analysis to understand the data and what features are useful for predicting churn
 - Feature Engineering to create features that will be used in the modelling process
 - Modelling using machine learning algorithms such as Logistic Regression, Random Forest, Gradient Boosted Trees, Linear SVM, Naive Bayes 

## File Descriptions
There is one exploratory notebook and html file of the notebook available here to showcase my work in predicting churn. Markdown cells were used throughout to explain the process taken.

## Medium Blog Post 
The main findings of the code can be found at the Medium Blog post available [here](https://stephirvine.medium.com/predicting-churn-with-pyspark-4c8edc8a19e0) explaining the technical details of my project.
A Random Forest Classifier was chosen to be the best model by evaluating F1 score and accuracy metrics. The final model achieved an F1 and Accuracy score of 0.88. 

## Licensing, Authors, Acknowledgements, etc.
I'd like to acknowledge Udacity for the project idea and workspace.

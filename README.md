# Loan-Default-Prediction
Machine learning model to predict if a customer will default their loan based on their transaction and load details.

How to run:
-----------
Clone the repository and run the LoadDefault.py

Requirements to run:
--------------------
1. numpy
2. matplotlib
3. pandas
4. sklearn
5. xgboost

About the model:
----------------
This prediction model uses the ‘Gradient Boosting Algorithm’ from xgboost. The usage of decision tree based classifier opens our model to various other applications like easy visualization of the model, deriving sensible features out of the model’s conditions and not limiting it to prediction alone.

Model evaluation:
-----------------
The performance of a classifier is best evaluated using the cross validation score and AUC of mean ROC curve. A score of ‘1’ in either of these evaluation means the model is 100% efficient. This model has gained 0.796 cross validation score, 0.765 mean ROC AUC and 0.782 validation set ROC AUC.

The customers who achieve a probability of 0.90 of loan default from this model are very likely to default on the loan and those who get a probability of less than 0.1 can be considered very safe.
# Telecom churn analysis and prediction

## Business problem overview
In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.

For many incumbent operators, retaining high profitable customers is the number one business goal.
To reduce customer churn, telecom companies need to predict which customers are at high risk of churn.
Here we will analyse customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn and identify the main indicators of churn.


## Definitions of churn
There are various ways to define churn, such as:

**Revenue-based churn:** Customers who have not utilised any revenue-generating facilities such as mobile internet, outgoing calls, SMS etc. over a given period of time. One could also use aggregate metrics such as ‘customers who have generated less than INR 4 per month in total/average/median revenue’.
The main shortcoming of this definition is that there are customers who only receive calls/SMSes from their wage-earning counterparts, i.e. they don’t generate revenue but use the services. For example, many users in rural areas only receive calls from their wage-earning siblings in urban areas.

**Usage-based churn:** Customers who have not done any usage, either incoming or outgoing - in terms of calls, internet etc. over a period of time.
A potential shortcoming of this definition is that when the customer has stopped using the services for a while, it may be too late to take any corrective actions to retain them. For e.g., if you define churn based on a ‘two-months zero usage’ period, predicting churn could be useless since by that time the customer would have already switched to another operator.

 ## Usage-based churn
In this project, we will use the usage-based definition to define churn.
Customers usually do not decide to switch to another competitor instantly, but rather over a period of time (this is especially applicable to high-value customers). In churn prediction, we assume that there are three phases of customer lifecycle :

**The ‘good’ phase:** In this phase, the customer is happy with the service and behaves as usual.

**The ‘action’ phase:** The customer experience starts to sore in this phase, for e.g. he/she gets a compelling offer from a competitor, faces unjust charges, becomes unhappy with service quality etc. In this phase, the customer usually shows different behaviour than the ‘good’ months. Also, it is crucial to identify high-churn-risk customers in this phase, since some corrective actions can be taken at this point (such as matching the competitor’s offer/improving the service quality etc.)

**The ‘churn’ phase:** In this phase, the customer is said to have churned. You define churn based on this phase. Also, it is important to note that at the time of prediction (i.e. the action months), this data is not available to you for prediction. Thus, after tagging churn as 1/0 based on this phase, you discard all data corresponding to this phase.

## Approach:

- We have performed data preprocessing, Missing Value Analysis, Feature Engineering, identified most valuable customers, tagged churners, and performed required EDA. We have mentioned few inferences observed during EDA.
- As part of data preparation, we have split the data into train-test dataset and performed SMOTE on training dataset to handle class imbalance. We have performed scaling (Used Robust scaling to handle outliers) before building our first model.
- We have used Logistic Regression model on actual features. We have used RFE to select top 20 features and then performed manual tunning to remove multicollinearity and make sure that all beta coefficients are statistically significant. From final Logistic Regression model, we can find below features importance to perform churn prediction.

![image](https://user-images.githubusercontent.com/77941537/142938559-a07bd650-d305-45ed-bdde-bb572bc9d76e.png)

  - We have got almost similar Accuracy and Recall on training and testing dataset.
  - Here probability cutoff .5 is well balanced. If business wants to increase Recall/Sensitivity further, then this probability cut-off can be reduced to .4 or .3, but in that     case specificity and precision will reduce further (More users will be identify as churners who are actually not).
- Then we have performed PCA and kept 97% of variance by selecting 39 Principal Components. We used Random Forest and then performed hyperparameters tunning to tune our Random Forest model. Final Classification on testing dataset report after performing hyperparameters tunning is as below:
![image](https://user-images.githubusercontent.com/77941537/142938522-7c4294a7-3329-4205-bae3-f82972c5b2c0.png)

- Followed by that, We used XGBoost Classifier and then perfomed hyperparameters tunning to tune our XGB Classifier model. Final Classification on testing dataset report after performing hyperparameters tunning is as below:
![image](https://user-images.githubusercontent.com/77941537/142938461-dadb5db6-a5ae-46de-b501-79c18d48ea34.png)

## Feature importance:
![image](https://user-images.githubusercontent.com/77941537/142938608-1530c60e-e943-41d4-8c76-7dec94262a09.png)

# Product-Delivery-Date-Prediction (Python, Google Colab)

This project focused on how to apply Machine Learning to build a model that can accurately predict delivery dates for items sold by a retailer using a dataset with details of shipping information. For our analysis, we decided to train and implement multiple machine learning models, consisting of Linear Regression, Random Forest, Gradient Boosting Model (GBM), Support Vector Machine (SVM), and a final ensemble model. To score the model, we will be looking at RMSE (Root-mean-square deviation).

Data Source
The dataset is taken from eBay from the ml challenge to estimate the delivery time. The data is provided with all the shipment and delivery related details and requires us to estimate the delivery date for items purchased. Features available were -

Acceptance_scan_timestamp, Declared_handling_days, Shipment_method_id, Carrier_min_estimate, Carrier_max_estimate, Item_zip, Buyer_zip, Category_id, Package_size

Data Pre-processing
<img width="1792" alt="DataPreprocessing" src="https://user-images.githubusercontent.com/78490598/164074686-5ef2dc40-1870-4e6d-80ec-4c15e0e7ed54.png">


Data Transformation
The delivery date, payment date, and acceptance date columns were converted into a numerical format based on the month, year, and day.
Columns with zip codes were converted to numeric values as well, but only the first five numbers of each zip code were used to avoid conflicts.
All the columns were scaled using the minmaxscaler from Sklearn into a range from 0 to 1 to aid the model with its computational power.
Feature Selection
Heatmap Feature Selection
<img width="1148" alt="Feature Selection" src="https://user-images.githubusercontent.com/78490598/164074748-87896dc6-5a19-4e61-9b1e-84ecf24e6166.png">


Model Comparison
As a regression problem, evaluation was done based on the Root mean squared error, mean absolute error, R2_score, and mean sum of squared error. Different models were implemented to figure out which model will be better suited to predict delivery date more accurately.

Based on the comparison among all the models, the ensemble method on random forest regression, gradient boost regression, and CatBoost regression outperforms all the others.

<img width="845" alt="ModelComparison" src="https://user-images.githubusercontent.com/78490598/164074891-757b9150-96ef-4b91-abb3-d83ccd12448e.png">


After Hyperparameter tuning
For the three models mentioned above, hyperparameter tuning was done to find the best parameter for higher accuracy. Used RandomizedSearchCV, GridSearchCV to fine-tune the models’ hyperparameters.

<img width="585" alt="AfterHyperparameter_modelcompare" src="https://user-images.githubusercontent.com/78490598/164074950-3611185a-7200-4d4b-9e05-ba89ef9a88a1.png">

After the hyper tuning step, Random Forest gave about 20% more r2_score than it did before hyper tuning with 2.54 MSE, 1.59 RMSE, and 0.97 MAE scores. Since the r2_score didn't change drastically, the ensemble approach was used to get majority votes from these selected models. R2_score 0.74 with MSE 2.53 was obtained using ensemble regression, which was about 0.1% better than random forest regressor.

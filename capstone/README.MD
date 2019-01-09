# Starbucks Capstone Project

## Project Description
This is my capstone project for the Udacity Data Scientists Nanodegree. This project involves simulated data that mimics customer behavior on the Starbucks rewards mobile app.

The data consists of customers' demographics attributes, promotions' details and timestamped customer transaction data.

There are 10 different types of offers available. Starbucks will send these offer to users of the mobile app on a weekly basis. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.

Through EDA, I noted that customers generally do not spend more than they normally do when given promotions. The EDA can be found in EDA.ipynb.

Hence, the goal of this project will be to implement a machine-learning based promotional strategy that identifies customers who are more likely to increase their expenditure if they are given promotions. The promotions will only be send to these promising customers. This strategy is commonly referred to as Uplift Modelling. The uplift model that I will be using for this project is just 1 of the many different types of uplift models available.

I will evaluate the strategy's performance by calculating the Net Incremental Revenue (NIR) of the promotional strategy. This will estimate the amount of additional profits for Starbucks should the strategy be implemented.

The NIR will be calculated from individuals who our model predicts should receive the promotions. We will then look at their actual transaction history to calculate the NIR. 

For example, our model predicts that Bob is a promising customer and shoud receive a discount promotion during the first month. The discount promotion has a difficulty of $10 (meaning he has to spend $10 to activate it), a reward of $2 and a validity period of 5 days (meaning he has to complete the offer with 5 days to activate the reward). Bob spends $11 to activate the promotion within its validity period. During the same month, Bob also spends a total of $5 during days when he did not received any promotion. The NIR will be:

NIR = promotional revenue - cost of promotions - nonpromotional revenue = $11 - $2 - $5 = $4

If the promotion is not activated, then there will be no cost.

For this project, each offer type will have its own individualized promotional strategy model. Customers' transactional behavior will be aggregated on a monthly basis.

Each offer will have 3 to 4 months of training data, 1 month of validation data and 1 month of test data.

The validation and test NIR will be reported.

The original dataset have a significant amount of misssing values for the demographics data. Machine-learning will be used to predict these missing values. The code for this section can be found in the input_missing_data.ipynb.

The data also requires substantial preprocessing. This will be performed in generate_monthly_data.ipynb.

The code for the uplift models for each offer can be found in the files Model_Offer_0.ipynb/.../Model_Offer_9.ipynb.

Part 1 of the project's write-up can be found [here](https://towardsdatascience.com/implementing-a-profitable-promotional-strategy-for-starbucks-with-machine-learning-part-1-2f25ec9ae00c)

Part 2 of the project's write-up can be found [here](https://towardsdatascience.com/implementing-a-profitable-promotional-strategy-for-starbucks-with-machine-learning-part-2-8dd82b21577c)

## File Description
~~~~~~~
        capstone
          |-- data
                |-- portfolio.json
                |-- profile.json
                |-- transcript.json
          |-- generate_monthly_data.ipynb
          |-- input_missing_data.ipynb
          |-- model_helper.py
          |-- Model_Offer_0.ipynb
          |-- Model_Offer_1.ipynb
          |-- Model_Offer_2.ipynb
          |-- Model_Offer_3.ipynb
          |-- Model_Offer_4.ipynb
          |-- Model_Offer_5.ipynb
          |-- Model_Offer_6.ipynb
          |-- Model_Offer_7.ipynb
          |-- Model_Offer_8.ipynb
          |-- Model_Offer_9.ipynb
          |-- preprocessing_helper.py
          |-- EDA.ipynb
          |-- new_profile.csv
          |-- monthly_data_rolling.csv
          
~~~~~~~
The j.son files
1. portfolio.json: Contains information about the 10 varieties of offers available through the app.
2. profile.json: Contains demographics information about the customers.
3. transcript.json: Contains timestamps of when purchases are made on the app as well as timestamps of when offer is received, viewed or completed.

The .ipynb and .py files
1. input_missing_data.ipynb: Use machine learning to predict and fill missing demographics information in profile.json. Running this file will produce new_profile.csv, which will combine the original demographics information and the predicted demographics information for the missing data.
2. generate_monthly_data.ipynb: Generates monthly_data_rolling.csv, which is the monthly time series summary of user transactional behavior.
3. Model_Offer_0.ipynb/.../Model_Offer_9.ipynb: Contains the promotional strategy for each of the 10 different offers that are available.
4. model_helper.py: Contains helper functions for Model_Offer_0.ipynb/.../Model_Offer_9.ipynb.
5. preprocessing_helper.py: Contains helper function for input_missing_data.ipynb and generate_monthly_data.ipynb
6. EDA.ipynb: Contains the exploratory data analysis for the project

The .csv files
1. new_profile.csv: Contains information from profile.json with missing values filled using machine learning.
2. monthly_data_rolling.csv: Monthly time-series summary of user transactional behavior.

## Summary of Results
The baseline strategy will be to send the offer to everyone in the dataset, which is identical to the approach used in the original experiment. The uplift model will be the machine-learning based promotion strategy implemented for this project. Our objective will be to improve on the baseline results.

#### Offer id 0: Discount promotion with a difficulty of $20, a reward of $5, and a validity period of 10 days
Baseline Strategy ~ Validation NIR: $108.70, Test NIR: -$4,889.48 <br />
Uplift Model ~ Validation NIR: $72.83, Test NIR: -$2163.47

#### Offer id 1: Discount promotion with a difficulty of $7, a reward of $3, and a validity period of 7 days
Baseline Strategy ~ Validation NIR: $185.14, Test NIR: -$4,732.18 <br />
Uplift Model ~ Validation NIR: $60.41, Test NIR: $4.61

#### Offer id 2: Discount promotion with a difficulty of $10, a reward of $2, and a validity period of 7 days
Baseline Strategy ~ Validation NIR: $65.88, Test NIR: -$5,519.62 <br />
Uplift Model ~ Validation NIR: $12.40, Test NIR: $3.17

#### Offer id 3: Informational promotion with a validity period of 4 days
Baseline Strategy ~ Validation NIR: -$4,193.67, Test NIR: -$8,754.95 <br />
Uplift Model ~ Validation NIR: $29.39, Test NIR: -$34.26

#### Offer id 4: Buy-one-get-one-free promotion with a difficulty of $10, a reward of $10 and a validity period of 5 days
Baseline Strategy ~ Validation NIR: -$4,634.69, Test NIR: -$7027.36 <br />
Uplift Model ~ Validation NIR: $12.39, Test NIR: $10.20

#### Offer id 5: Informational promotion with a validity period of 5 days
Baseline Strategy ~ Validation NIR: -$5,188.06, Test NIR: -$6,707.87 <br />
Uplift Model ~ Validation NIR: $2.19, Test NIR: -$130.91

#### Offer id 6: Buy-one-get-one-free promotion with a difficulty of $5, a reward of $5 and a validity period of 7 days.
Baseline Strategy ~ Validation NIR: $121.58, Test NIR: -$6,542.62 <br />
Uplift Model ~ Validation NIR: $21.81, Test NIR: $10.15

#### Offer id 7: Buy-one-get-one-free promotion with a difficulty of $10, a reward of $10 and a validity period of 7 days.
Baseline Strategy ~ Validation NIR: $65.13, Test NIR: -$6,207.28 <br />
Uplift Model ~ Validation NIR: $24.29, Test NIR: $0.73

#### Offer id 8: Buy-one-get-one-free promotion with a difficulty of $5, a reward of $5 and a validity period of 5 days.
Baseline Strategy ~ Validation NIR: -$5,779.91, Test NIR: -$7,508.97 <br />
Uplift Model ~ Validation NIR: $481.78, Test NIR: -$786.3

#### Offer id 9: Discount promotion with a difficulty of $10, a reward of $2 and a validity period of 10 days
Baseline Strategy ~ Validation NIR: $104.30, Test NIR: -$5,006.65 <br />
Uplift Model ~ Validation NIR: $51.87, Test NIR: $3.02

## Instructions
1. Run input_missing_data.ipynb to generate new_profile.csv.
2. Run generate_monthly_data.ipynb to generate monthly_data_rolling.csv.
3. Run EDA.ipynb to view the expolratory data analysis.
4. Run Model_Offer_0.ipynb/ ... /Model_Offer_9.ipynb for each of the 10 promotion strategies catered to each offer type.

## Installations
Anaconda, XGBoost, Imblearn, json, pickle

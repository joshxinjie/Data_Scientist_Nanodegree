# Starbucks Capstone Project

## Project Description
This is my capstone project for the Udacity Data Scientists Nanodegree. This project involves simulated data that mimics customer behavior on the Starbucks rewards mobile app. 

The goal of this project will be to implement a machine-learning based promotional strategy that identifies customers who are more likely to increase their expenditure when they are given a promotion as opposed to when they are not given a promotion. In addition, I will devise a testing methodology to evaluate the strategy's performance in real-life by calculating the Net Incremental Revenue (NIR) of the promotional strategy. This will estimate the amount of additional profits for Starbucks should the strategy be implemented.

There are 10 different types of offers available. Starbucks will send these offer to users of the mobile app on a weekly basis. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.

For this project, each offer type will have its own individualized promotional strategy model. Customers' transactional behavior will be aggregated on a monthly basis. The first 15 months are be used as training data for the promotional strategy. Months 16 and 17 will be the validation months, and they will be used to tune the promotional strategies through grid search. Months 18 and 19 will serve as the tests months, and they will act as the final check of our promotional strategies.

This proved to be a relatively challenging problem as evident by the results obtained.

The detailed write-up of the project can be found at the accompanying blog [post](https://medium.com/@joshxinjielee/designing-and-testing-a-promotion-strategy-for-starbucks-2f25ec9ae00c)

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
3. model_helper.py: Contains helper functions for Model_Offer_0.ipynb/.../Model_Offer_9.ipynb.
4. Model_Offer_0.ipynb/.../Model_Offer_9.ipynb: Contains the promotional strategy for each of the 10 different offers that are available.

The .csv files
1. new_profile.csv: Contains information from profile.json with missing values filled using machine learning.
2. monthly_data_rolling.csv: Monthly time-series summary of user transactional behavior.

## Summary of Results
The baseline strategy will be to send the offer to everyone in the dataset. The baseline results will serve as a benchmark that we will improve on. The uplift model will be the machine-learning based promotion strategy implemented for this project.

#### Offer id 0: Discount promotion with a difficulty of $20, a reward of $5, and a validity period of 10 days
Baseline Strategy ~ Validation NIR: -$5,268.54, Test NIR: -$4,889.48 <br />
Uplift Model ~ Validation NIR: $30.06, Test NIR: -$74.16

#### Offer id 1: Discount promotion with a difficulty of $7, a reward of $3, and a validity period of 7 days
Baseline Strategy ~ Validation NIR: -$6,168.08, Test NIR: -$4,732.18 <br />
Uplift Model ~ Validation NIR: $18.77, Test NIR: $0

#### Offer id 2: Discount promotion with a difficulty of $10, a reward of $2, and a validity period of 7 days
Baseline Strategy ~ Validation NIR: -$5,600.92, Test NIR: -$5,519.62 <br />
Uplift Model ~ Validation NIR: $10.36, Test NIR: $0

#### Offer id 3: Informational promotion with a validity period of 4 days
Baseline Strategy ~ Validation NIR: -$4,193.67, Test NIR: -$8,754.95 <br />
Uplift Model ~ Validation NIR: $45.96, Test NIR: $0

#### Offer id 4: Buy-one-get-one-free promotion with a difficulty of $10, a reward of $10 and a validity period of 5 days
Baseline Strategy ~ Validation NIR: -$4,634.69, Test NIR: -$7027.36 <br />
Uplift Model ~ Validation NIR: $31.63, Test NIR: -$37.27

#### Offer id 5: Informational promotion with a validity period of 5 days
Baseline Strategy ~ Validation NIR: -$5,188.06, Test NIR: -$6,707.87 <br />
Uplift Model ~ Validation NIR: $25.78, Test NIR: -$24.7

#### Offer id 6: Buy-one-get-one-free promotion with a difficulty of $5, a reward of $5 and a validity period of 7 days.
Baseline Strategy ~ Validation NIR: -$4,558.54, Test NIR: -$6,542.62 <br />
Uplift Model ~ Validation NIR: $32.10, Test NIR: -$88.43

#### Offer id 7: Buy-one-get-one-free promotion with a difficulty of $10, a reward of $10 and a validity period of 7 days.
Baseline Strategy ~ Validation NIR: -$4,842.09, Test NIR: -$6,207.28 <br />
Uplift Model ~ Validation NIR: $17.13, Test NIR: -$856.38

#### Offer id 8: Buy-one-get-one-free promotion with a difficulty of $5, a reward of $5 and a validity period of 5 days.
Baseline Strategy ~ Validation NIR: -$5,779.91, Test NIR: -$7,508.97 <br />
Uplift Model ~ Validation NIR: $502, Test NIR: -$38.95

#### Offer id 9: Discount promotion with a difficulty of $10, a reward of $2 and a validity period of 10 days
Baseline Strategy ~ Validation NIR: -$3,927.56, Test NIR: -$5,006.65 <br />
Uplift Model ~ Validation NIR: $52.11, Test NIR: -$60.37

While we are able to optimize the promotional strategies to achieve positive NIRs during the validation months, the current strategies do not produce positive results during the test months. I will hope to improve the strategies' performance in the future through further refinements.

## Instructions
1. Run input_missing_data.ipynb to generate new_profile.csv.
2. Run generate_monthly_data.ipynb to generate monthly_data_rolling.csv.
2. Run Model_Offer_0.ipynb/ ... /Model_Offer_9.ipynb for each of the 10 promotion strategies catered to each offer type.

## Installations
Anaconda, XGBoost, Imblearn, json, pickle

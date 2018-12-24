# Starbucks Capstone Project

## Project Description
This is my capstone project for the Udacity Data Scientists Nanodegree. This project involves simulated data that mimics customer behavior on the Starbucks rewards mobile app. 

The goal of this project will be to implement a machine-learning based promotional strategy that identifies customers who are more likely to increase their expenditure when they are given a promotion as opposed to when they are not given a promotion. In addition, I will devise a testing methodology to evaluate the strategy's performance in real-life by calculating the Net Incremental Revenue (NIR) of the promotional strategy. This will estimate the amount of additional profits for Starbucks should the strategy be implemented.

There are 10 different types of offers available. Starbucks will send these offer to users of the mobile app on a weekly basis. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.

For this project, each offer type will have its own individualized promotional strategy model.

The accompanying blog post can be found [here](https://medium.com/@joshxinjielee/designing-and-testing-a-promotion-strategy-for-starbucks-2f25ec9ae00c)

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

## Instructions
1. Run input_missing_data.ipynb to generate new_profile.csv.
2. Run generate_monthly_data.ipynb to generate monthly_data_rolling.csv.
2. Run Model_Offer_0.ipynb/ ... /Model_Offer_9.ipynb for each of the 10 promotion strategies catered to each offer type.

## Installations
Anaconda, XGBoost, Imblearn, json, pickle

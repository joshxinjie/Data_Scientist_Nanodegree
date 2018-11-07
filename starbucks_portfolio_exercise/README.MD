# Starbucks Portfolio Exercise

## Project Description
This portfolio exercise was originally used as a take-home assignment provided by Starbucks for their job candidates. This exercise involves an advertising promotion that was tested to see if it would bring more customers to purchase a specific product priced at $10. It costs the company 0.15 to send out each promotion. Ideally, we want to limit that promotion only to those that are most receptive to the promotion.

Our promotion strategy will be evaluated on 2 key metrics
1. Incremental Response Rate (IRR)
2. Net Incremental Revenue (NIR)

There are numerous approaches to tackle this problem. I have chosen to implement uplift models to optimize the promotional strategy. 4 different types of uplift models were implemented:

1. Traditional Uplift Model
2. Two Models Approach
3. SIngle Model with Treatment Indicator Variable
4. Four Quadrant Approach

The article for this project can be found [here](https://medium.com/datadriveninvestor/simple-machine-learning-techniques-to-improve-your-marketing-strategy-demystifying-uplift-models-dc4fb3f927a2).

## File Description
1. training.csv: Contains training data
2. Test.csv: Contains test data
3. Starbucks.ipynb: Contains code for the promotional strategy based on uplift models
4. test_results.py: Contains functions to evaluate the IRR and NIR values. File is provided by Starbucks.

## Instructions
1. Run Starbucks.ipynb

## Installations
Anaconda, XGBoost, Imblearn, Statsmodels, Seaborn

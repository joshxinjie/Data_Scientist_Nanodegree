# Write a Data Science Blog

## Description
In this project, I will investigate why certain Airbnb listings are more successful than others. This analysis will focus on Airbnb listings in Boston and Seattle. Estimated revenues of listings will also be used to define the success of a listing. The primary questions that will be answered in this project are:

1. How much revenue do Airbnb hosts make?
2. What are the best types of property to rent?
3. When is the best time to rent?
4. Which are the best areas to rent?
5. What should you write in a listing name to attract more attention?

In addition, several models (Linear Models, XGBoost) were trained to predict the estimated revenues of listings. The results of the models will be used to explore which are the most important factors in achieving higher revenues.

The article detailing the key findings of this project can be found [here](https://medium.com/@joshxinjielee/uncovering-the-secrets-to-higher-airbnb-revenues-6dbf178942af).

## File Descriptions
1. boston_airbnb_analysis.ipynb: Contains all the analysis for the Boston Airbnb datasets
2. seattle_airbnb_analysis.ipynb: Contains all the analysis for the Seattle Airbnb datasets

## Data
The data used in this project are the Boston Airbnb Open Data and the Seattle Airbnb Open Data from kaggle. They can be downloaded from this [link](https://www.kaggle.com/airbnb/boston), as well as this [link](https://www.kaggle.com/airbnb/seattle).

Put the data files in the following folder structure
~~~~~~~
        data_science_blog
          |-- boston_airbnb_analysis.ipynb
          |-- seattle_airbnb_analysis.ipynb
          |-- boston-airbnb-open-data
               |-- calendar.csv
               |-- listings.csv
               |-- reviews.csv
          |-- seattle
               |-- calendar.csv
               |-- listings.csv
               |-- reviews.csv
~~~~~~~

## Installations
Anaconda, Seaborn, Nltk, Re, WordCloud, XGBoost, Shap, Bokeh

To run the section of the code involving Bokeh, you need to get a Geocoding API from google [here](https://developers.google.com/maps/documentation/geocoding/get-api-key)

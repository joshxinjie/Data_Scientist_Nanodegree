# Identify Customer Segment

## Description
Analyze real-life data concerning a company that performs mail-order sales in Germany. The data is provided by Bertelsmann partners AZ Direct and Arvato Financial Solution. Demographics data from the mail-order firm's customers (191,652 individuals) and a subset of the German population (891,211 individuals) are provided for this analysis. In total, there are 85 demographic features available. 

The goal of the project is to apply unsupervised learning techniques to identify segments of the German population that are popular with the mail-order firm.

The resulting cluster analysis can be used for tasks such as identifying which facets of the population are likely to purchase the firm's products for a mailout campaign. 

## Data
1. Udacity_AZDIAS_Subset.csv: Demographic data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
2. Udacity_CUSTOMERS_Subset.csv: Demographic data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
3. Data_Dictionary.md: Information file about the features in the provided datasets.
4. AZDIAS_Feature_Summary.csv: Summary of feature attributes for demographic data.

Due to agreements with Bertelsmann, the datasets cannot be shared or used for used for tasks other than this project. Files have to be removed from the computer within 2 weeks after completion of the project.

## Methodology
1. Start with analyzing demographics data of general population. Deal with missing values in the dataset.
2. Drop variables with extremely high frequencies of missing values.
3. Drop rows/individuals with high amount of missing values. Noted that these individuals have relatively different distribution of data values on columns that are not missing data as compared to the rest of the population.
4. Re-encode categorical features to dummy or one-hot encoded features.
5. Identify mixed-type features and engineer new features from them. A mixed-type feature generally tracks information on two or more dimensions. For example, "CAMEO_INTL_2015" variable combines information on two axes: wealth and life stage. In this case, we will generate two new features: one tracking wealth and the other tracking life stage.
6. Perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Use StandardScaler to ccale each feature to mean 0 and standard deviation 1.
7. Perform dimensionality reduction using PCA. Identify which demographics attribute are the most positively or negatively correlated with each principal component. This will allow us to interpret the principal components.
8. Apply clustering to general population with KMeans.
9. Repeat steps 1-8 on the demographics data of the firm's customer, using the same StandardScaler, PCA and KMeans objects fitted to the general population.
10. Analyze the cluster distributions of the general population and the firm's customers. Identify clusters that are overrepresented or underrepresented in the customer dataset compared to the general population.
11. Identify what kinds of people are typified by these overrepresented or underrepresented clusters. For each cluster, identify which principal components have the highest means and isolate them. Using the interpretations of the principal components from step 7, analyze what demographics attributes the top principal components are capturing. Use the information to deduce the demographic attributes of the overrepresented and underrepresented clusters.

## Summary of Results
The mail-order company is extremely popular with individuals who:
 - have high-income
 - have high purchasing power
 - are married
 - live in more desirable neighburhoods
 - have an affinity for avant garde movement
 - supports environmental sustainability movements
 - Internet-savvy

The mail-order company is less popular with these groups (A to G) of individuals:

  Group A:
  - elderly
  - prefers shopping at brick and mortar stores as opposed to online shopping
  - more financially inconspicuous
  - more religious, rational, dutiful, traditional-minded and cultural-minded

  Group B:
  - low-income home renters
  - less interested financially
  - live in households with larger family sizes

  Group C:
  - low-income couples with no children
  - more materialistic and health conscious
  - more risk-averse with their insurance purchases
  - live in buildings that are more densely occupied and contain more households with larger family sizes

  Group D:
  - are financially inconspicuous
  - are foreign born
  - live in less desirable neighbourhoods
  - prioritize low cost when purchasing energy
  - more likely to be materialistic, rational and traditional-minded

  Group E:
  - middle class individuals who live in rural areas
  - work as farmers or in occupations that support the farming community
  - high purchasing power in the area they live in
  - reside in areas that are located in what was formerly known as East Germany

  Group F:
  - middleclass family with young or teenage children as well as single-parents and couples without children
  - live in multi-generational or multi-person household

  Group G:
  - demanding shoppers
  - consume a lot of traditional (not online) advertisments or no advertisements at all
  - low-income earners who have little aspirations of moving up the income ladder, or are financially stable and own homes
  - financially prepared and like to invest
  - like going on vacation for cultural sightseeing and/or participating in package tours
  - less likely to travel within their home country
  - recent immigrants.
  - live in more desirable neighbourhoods

This mail-order company is indifferent to this group of individuals who:
 - are young unmarried males
 - have high-income
 - prefers the use of green energy
 - enjoys shopping for fun, entertainment and satisfaction
 - less family-minded, socially-minded and cultural-minded
 - less religious and dreamful
 - more rational, dominant-minded and critical-minded

## How To Run It
Run identify_customer_segments.ipynb

## Installations
Anaconda, Seaborn

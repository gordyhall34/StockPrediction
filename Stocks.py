# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 20:56:17 2022

@author: james

"""


#%%
import numpy as np
import pandas as pd #this imports the excel sheet to be used for the program

def load_stock_data():
    csv_path = pd.read_csv (r'C:\Users\james\OneDrive\Documents\09-ECEC 612\02-HW2\01-code\stock.csv')
    return csv_path

#%%

stock = load_stock_data() #associates the data with a variable
stock_head = stock.head() #shows the top 5 rows of the data sheet
#print("\n",stock_head)

#%%
stock_info = stock.info() #shows the meta-data of the file, including type of data and amount of each
stock_descrip = stock.describe() #Shows a summary of the float64 attributes
#print("\n", stock_descrip)
#%%
stock["Close"] = pd.to_numeric(stock.Close, errors = "coerce")

stock_close = stock[["Close"]]
stock_close.head(30)
#print(stock_close.head(30))
#shows first 30 instances of close data
stock["Close"].fillna(int(stock["Close"].median()),inplace = True)

#%%
from sklearn.preprocessing import OneHotEncoder

#imports onehotencoder
cat_encoder = OneHotEncoder()
stock_cat_1hot = cat_encoder.fit_transform(stock_close)
#converts non numeric to one hot values
#print(stock_cat_1hot)
#%%
stock_cat_1hot.toarray()
#converts to a 2D array
#print(stock_cat_1hot.toarray())
#print(cat_encoder.categories_)
#%%
stock["Low"] = pd.to_numeric(stock.Low, errors = "coerce")

stock_low = stock[["Low"]]

#imports onehotencoder
cat_encoder = OneHotEncoder()
stock_cat_1hot = cat_encoder.fit_transform(stock_low)
#converts non numeric to one hot values

stock_cat_1hot.toarray()

#%%
stock["Volume"] = pd.to_numeric(stock.Volume, errors = "coerce")

stock_volume = stock[["Volume"]]

#imports onehotencoder
cat_encoder = OneHotEncoder()
stock_cat_1hot = cat_encoder.fit_transform(stock_volume)
#converts non numeric to one hot values

stock_cat_1hot.toarray()


####################################################
#stock_num = stock.drop("Date", axis=1)
#removes every data type except for flat64
stock_num = stock.select_dtypes(include=['float64'])

#%%
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
#creates an imputer that uses the median strategy to fill empty values

#%%
imputer.fit(stock_num)
#creates median values of the data types in stock_num 

#%%
imputer.statistics_
#stores the median values in the stats instance variable
stock_num.median().values

#print("Stats \n",imputer.statistics_)
#print("Stock median values \n",stock_num.median().values)

#%%
X = imputer.transform(stock_num)
#replaces missing values with the median

#%%
stock_tr = pd.DataFrame(X, columns = stock_num.columns, index = stock_num.index)
#print(stock_tr)
#places array back into dataframe

#%%
import matplotlib.pyplot as plt 
stock.hist(bins=50, figsize=(20,15)) #creates a histogram of numerical data 
plt.show() #displays the histogram



#%%
from sklearn.model_selection import train_test_split
#creates a test set of data using 20% of data set, and a random num generator seed of 42
train_set, test_set = train_test_split(stock, test_size=0.2, random_state=42)
#test_set = print(test_set.head()) #prints test set head, wanted to see if it worked

#%%
stock["Close"].hist() #shows the histogram of the opening prices

#%%
stock["close_cat"] = pd.cut(stock["Close"], # redistributes the open histogram
                               bins=[0., 50, 200, 250, 350, np.inf],
                               labels=[1, 2, 3, 4, 5])

#%%
print(stock["close_cat"].value_counts()) #shows new distribution, for checking

#%%

stock["close_cat"].hist() #displays new histogram

#%%
from sklearn.model_selection import StratifiedShuffleSplit 
#creates a stratified shuffle test set using the new open_cat distribution
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(stock, stock["close_cat"]):
    strat_train_set = stock.loc[train_index]
    strat_test_set = stock.loc[test_index]
    
#%%
strat_test_valCount = strat_test_set["close_cat"].value_counts() / len(strat_test_set)
#print(strat_test_valCount)
#shows the catefory proportions in strat test set
#%%
stock_val_count = stock["close_cat"].value_counts() / len(stock)
#print(stock_val_count)
#shows the category propotions for the full set of data
#%%
for set_ in (strat_train_set, strat_test_set):
    set_.drop("close_cat", axis=1, inplace=True)
    #removes the close_cat to return data to original state
    
stock = strat_train_set.copy()
# makes a copy of training data set so the origial training set isnt damaged
#%%

corr_matrix = stock.corr()
corr_matrix["High"].sort_values(ascending=False)
print("\n")
print(corr_matrix)
# creates correlation matrix of the High price data
#%%
from pandas.plotting import scatter_matrix
attributes = ["Open", "High","Low","Close","Volume"]
scatter_matrix(stock[attributes], figsize=(12, 8))
#creates a daat plot comparing open to high prices of stock 
#%%
stock.plot(kind="scatter", x="Open", y="High",
	      alpha=0.1)
#shows just the scatter plot

#%%
corr_matrix = stock.corr()

#print(corr_matrix["Open"].sort_values(ascending=False))
#shows the correlation between the numerica attributes

#%%
stock["high_to_close"] = stock["Close"]/stock["High"]
corr_matrix = stock.corr()
#print(corr_matrix["Open"].sort_values(ascending = False))

#%%
attributes = ["High", "Close"]
scatter_matrix(stock[attributes], figsize=(12, 8))
stock.plot(kind="scatter", x="High", y="Close",
	      alpha=0.1)
#shows just the scatter plot

#%%

#sample_incomplete_rows = stock[stock.isnull().any(axis=1)]
#print(sample_incomplete_rows)

#%%
#stock = strat_train_set.drop("High", axis=1) 
stock_labels = strat_train_set["High"].copy()
#print("\n", stock_labels)
#shows the labels for the stock data set

#%%

#%%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        #('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

stock_num_tr = num_pipeline.fit_transform(stock_num)
#scaler pipeline for numeric values

#%%
from sklearn.compose import ColumnTransformer

stock.drop("high_to_close", axis = 1)
num_attribs = list(stock_num)
cat_attribs = ["Low","Close","Volume"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
stock_prepared = full_pipeline.fit_transform(stock)
#print(stock_prepared)
#print(stock.Volume)
#performs scaling for both numerical and categrical values in dataset

#%%
from sklearn.linear_model import LinearRegression
#import linearr regression model
lin_reg = LinearRegression()
lin_reg.fit(stock_prepared, stock_labels)

#%%
some_data = stock.iloc[:5]
#gets data from dataset
some_labels = stock_labels.iloc[:5]
 #gets labels from dataset
some_data_prepared = full_pipeline.transform(some_data)
#prepares the data to be used in predicitons
#%%
print("\n")

print("Lin Predictions:", lin_reg.predict(some_data_prepared))


#%%
print("\n")

print("Lin Labels:", list(some_labels))

#%%
from sklearn.metrics import mean_squared_error

stock_predictions = lin_reg.predict(stock_prepared)
lin_mse = mean_squared_error(stock_labels, stock_predictions)
lin_rmse = np.sqrt(lin_mse)

lin_rmse
print("Linear RMSE: ",lin_rmse)
#%%
from sklearn.model_selection import cross_val_score
lin_scores_val = cross_val_score(lin_reg, stock_prepared, stock_labels,
                         scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores_val = np.sqrt(-lin_scores_val)
#measures the RMSE of the linear regression model 


#%%
def display_scores(lin_rmse_scores_val):
     print("Lin Scores:", lin_rmse_scores_val)
     print("Lin Mean:", lin_rmse_scores_val.mean())
     print("Lin Cross val Standard deviation:", lin_rmse_scores_val.std())
     
display_scores(lin_rmse_scores_val)
#displays the scores for the linear model 
#%%
"""
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(stock_prepared, stock_labels)
#creates a decision tree regressor model 
#%%
stock_predictions = tree_reg.predict(stock_prepared)
tree_mse = mean_squared_error(stock_labels, stock_predictions)
tree_rmse = np.sqrt(tree_mse)

tree_rmse
print("tree RMSE",tree_rmse)
#measures RMSE
#%%
from sklearn.model_selection import cross_val_score
tree_scores = cross_val_score(tree_reg, stock_prepared, stock_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
#performs cross validation evaluatons
#%%

def display_scores(tree_scores):
     print("Tree Scores:", tree_scores)
     print("Tree Mean:", tree_scores.mean())
     print("Tree Standard deviation:", tree_scores.std())
     #prints the scores of the tree model
#%%
display_scores(tree_rmse_scores)

#%%
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()

forest_reg.fit(stock_prepared, stock_labels)

stock_predictions = forest_reg.predict(stock_prepared)
forest_rmse = mean_squared_error(stock_labels, stock_predictions)
forest_rmse = np.sqrt(forest_rmse)
print(forest_rmse)
#imports forest model 
#%%
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, stock_prepared, stock_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

#%%
forest_rmse_scores

#%%

X_test = strat_test_set.drop("High", axis=1)
print(X_test)
y_test = strat_test_set["High"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = forest_reg.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
"""


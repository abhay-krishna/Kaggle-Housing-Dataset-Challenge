import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

correlations = train_data.select_dtypes(include = [np.number]).corr()

top_positive_corr = correlations['SalePrice'].sort_values(ascending = False)[1:11]   #finding 5 most positively correlated variables

##print(positive_corr)

##plot.scatter(x=train.GrLivArea, y=np.log(SalePrice))
##plot.show()

#Removing outliers from top postively correlated features
train_data = train_data[train_data['GrLivArea'] < 4000]
train_data = train_data[train_data['GarageArea'] < 1100]
train_data = train_data[train_data['TotalBsmtSF'] < 2470]
train_data = train_data[train_data['1stFlrSF'] < 2750]


non_num_features = train_data.select_dtypes(include = 'object')
#print(non_num_features.describe())

#Feature engineering of training data

#Street type
train_data['street_numeric'] = pd.get_dummies(train_data.Street, drop_first = True)

#Alley type
train_data['alley_type_numeric'] = train_data.Alley.apply(lambda val: 1 if val == "Pave" else 0)

#Utilities
train_data['utilities_numeric'] = train_data.Utilities.apply(lambda val: 1 if val == "AllPub" else 0)
 
#Sale condition
train_data['sale_cond_numeric'] = train_data.SaleCondition.apply(lambda val: 1 if val == "Partial" else 0)

#Sale Type
train_data['sale_type_numeric'] = train_data.SaleType.apply(lambda val: 1 if val == "WD" else 0)

#Central air-conditioning
train_data['central_air_numeric'] = train_data.CentralAir.apply(lambda val: 1 if val == "Y" else 0)

#Roof style
train_data['roof_numeric'] = train_data.RoofStyle.apply(lambda val: 1 if val == "Shed" else 0)

#Lot shape
train_data.LotShape = train_data.LotShape.replace({'Reg':1, 'IR1':2, 'IR2':3, 'IR3':4})

#Proximity to various conditions
train_data['cond1_numeric'] = train_data.Condition1.apply(lambda val: 1 if val == "PosA" else 0)

#Kitchen quality
train_data['kitchenqual_numeric'] = train_data.KitchenQual.replace({'Ex':5,'Gd':4,'TA':3, 'Fa':2, 'Po':1})

#Condition for new house
train_data['new_house'] = train_data.apply(lambda x: 1 if x['YearBuilt'] == x['YrSold'] else 0, axis = 1)

nn_train_data = train_data.select_dtypes(include = [np.number]).interpolate('zero').dropna()  #Obtaining numerical training data with non-missing values

#Feature engineering of test data - identical to training data
test_data['street_numeric'] = pd.get_dummies(test_data.Street, drop_first = True)
test_data['alley_type_numeric'] = test_data.Alley.apply(lambda val: 1 if val == "Pave" else 0)
test_data['utilities_numeric'] = test_data.Utilities.apply(lambda val: 1 if val == "AllPub" else 0)
test_data['sale_cond_numeric'] = test_data.SaleCondition.apply(lambda val: 1 if val == "Partial" else 0)
test_data['sale_type_numeric'] = test_data.SaleType.apply(lambda val: 1 if val == "WD" else 0)
test_data['central_air_numeric'] = test_data.CentralAir.apply(lambda val: 1 if val == "Y" else 0)
test_data['roof_numeric'] = test_data.RoofStyle.apply(lambda val: 1 if val == "Shed" else 0)
test_data['cond1_numeric'] = test_data.Condition1.apply(lambda val: 1 if val == "PosA" else 0)
test_data.LotShape = test_data.LotShape.replace({'Reg':1, 'IR1':2, 'IR2':3, 'IR3':4})
test_data['kitchenqual_numeric'] = test_data.KitchenQual.replace({'Ex':5,'Gd':4,'TA':3, 'Fa':2, 'Po':1})
test_data['new_house'] = train_data.apply(lambda x: 1 if x['YearBuilt'] == x['YrSold'] else 0, axis = 1)

nn_test_data = test_data.select_dtypes(include = [np.number]).drop(['Id'], axis = 1).interpolate('zero').dropna()

#Model Building
x = nn_train_data.drop(["Id","SalePrice"], axis = 1)    #excluding the target variable and unnecessary columns
y = np.log(train_data.SalePrice)    #taking log to reduce the skew or asymmetry

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.25)   #Partitioning training dataset

train_model = Ridge(alpha = 6).fit(x_train, y_train) #building Ridge regression model and fitting it with training set

#Testing phase
log_output = train_model.predict(nn_test_data)

##log_output = train_model.predict(x_test)
##print("RMSE is ",mean_squared_error(y_test, predictions))

dollar_output = np.exp(log_output)    #converting from log-space to dollars

#Building final dataframe
solution = pd.DataFrame()
solution["Id"] = test_data.Id
solution["SalePrice"] = dollar_output

#Converting dataframe to CSV file
solution.to_csv('solution.csv', index = False)

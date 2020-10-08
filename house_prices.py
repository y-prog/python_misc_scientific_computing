#! /usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import xlrd
import time

start_time = time.time()



#C:\Users\digiovanniyani\Downloads


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import xlrd


train_data=pd.read_csv(r'/home/ydg/Desktop/csv_files/train.csv')


train_data_list=  train_data.values.tolist()

list1=list(train_data.shape)
rows=list1[0]
cols=list1[1]

my_array=np.array(train_data_list)
shape=(rows,cols)

my_array.reshape(shape)


def column(matrix, i):
    return [row[i] for row in matrix]

#print (type(column(my_array,2)[1]))
#print(type(column(my_array,6)[1]))


train_data_list=  train_data.values.tolist()
#train_data_list.append(train_data)   #  =train_data.to_numpy()
list1=list(train_data.shape)
rows=list1[0]
cols=list1[1]

my_array=np.array(train_data_list)
shape=(rows,cols)

my_array.reshape(shape)


def column(matrix, i):
    return [row[i] for row in matrix]

#print (type(column(my_array,2)[1]))
#print(type(column(my_array,6)[1]))

a=(train_data['GarageYrBlt'].mode())
print(type(a))

"""def file_values(x):
    counterlist = []
    newlist=[]
    k = 0
    my_counter = 0
    for i in range(cols):
        for j in range(rows):
            k +=1
            if column(my_array, i)[j] == x:
                my_counter += 1
                if k % rows == 0:
                    counterlist.append(my_counter)

                    if my_counter > 1000:
                        newlist.append(list(train_data.columns)[i])
                        break
    return(newlist)



null_values='nan'
print((file_values(null_values)))  #cols names with more than 1000 empty entries
"""
null_values=['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']

train_data.drop(null_values,axis=1,inplace=True)

#print(train_data.head())



train_data['LotFrontage']=train_data['LotFrontage'].fillna(train_data['LotFrontage'].mean())
train_data['BsmtCond']=train_data['BsmtCond'].fillna('TA')
train_data['BsmtQual']=train_data['BsmtQual'].fillna('TA')
train_data['BsmtExposure']=train_data['BsmtExposure'].fillna('No')
train_data['BsmtFinType1']=train_data['BsmtFinType1'].fillna('Unf')
train_data['BsmtFinType1']=train_data['BsmtFinType1'].fillna('Unf')
train_data['BsmtFinType2']=train_data['BsmtFinType2'].fillna('Unf')

train_data['GarageType']=train_data['GarageType'].fillna('Attchd')
train_data['GarageFinish']=train_data['GarageFinish'].fillna('Unf')
train_data['GarageQual']=train_data['GarageQual'].fillna('TA')
train_data['GarageCond']=train_data['GarageCond'].fillna('TA')

train_data['GarageYrBlt']=train_data['GarageYrBlt'].fillna(2005)

#print(train_data.shape)
#print(train_data.info())
print('====================================================================')

print('the mode is =-==--=-=-=-=-=-=-=-=-=-',train_data['MasVnrType'].mode())
print('the mode is =-==--=-=-=-=-=-=-=-=-=-',train_data['MasVnrArea'].mode())
print('the mode is =-==--=-=-=-=-=-=-=-=-=-',train_data['Electrical'].mode())
print('the mode is =-==--=-=-=-=-=-=-=-=-=-',train_data['GarageCond'].mode())
#print(train_data['BsmtFinType1'].head(100).to_string())


#sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False) #check remaining null values

#train_data['GarageYrBlt'].value_counts().plot(kind='bar');

plt.show()


null_cols=train_data.columns[train_data.isnull().any()]
#print(train_data[null_cols].isnull().sum())            #print cols with n null values

pd.set_option('display.max_columns', None)
numeric_features = train_data.select_dtypes(include=[np.number])
#print(numeric_features)

corr = numeric_features.corr()
print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])

categoricals = train_data.select_dtypes(exclude=[np.number])  #categorical features
#print(categoricals)

print(train_data.Street.value_counts())
train_data['enc_Street']=pd.get_dummies(train_data.Street, drop_first=True)
print(train_data.enc_Street.value_counts())

condition_pivot = train_data.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


print(pd.pivot_table(train_data, index = 'SaleCondition', values="SalePrice")) #check features related to sales price

def encode(x):
     return 1 if x == 'Partial' else 0 # 'partial' has the highest price
train_data['SaleCondition']=train_data['SaleCondition'].apply(encode)

def encode_1(x):
     return 1 if x == 'FV' or x == 'RL' else 0 # 'FV' has the highest price
train_data['MSZoning']=train_data['MSZoning'].apply(encode)
    

def encode_1(x):
     return 1 if x == 'Pave' else 0 # 'pave' has the highest price
train_data['Street']=train_data['Street'].apply(encode)

def encode_1(x):
     return 1 if x == 'IR2' or x == 'IR3' else 0 #feature highest price return 1
train_data['LotShape']=train_data['LotShape'].apply(encode)

def encode_1(x):
     return 1 if x == 'HLS' or x == 'Low' else 0 #feature highest price return 1
train_data['LandContour']=train_data['LandContour'].apply(encode)

def encode_1(x):
     return 1 if x == 'AllPub' else 0 #feature highest price return 1
train_data['Utilities']=train_data['Utilities'].apply(encode)

def encode_1(x):
     return 1 if x == 'CulDSac' or x == 'FR3' else 0#feat highest price return 1
train_data['LotConfig']=train_data['LotConfig'].apply(encode)


data.drop("LandSlope", axis=1, inplace=True)

def encode_1(x):
     return 1 if x == 'NoRidge' or x == 'NridgHt' or x == 'StoneBr' else 0#feat highest price return 1
train_data['Neighborhood']=train_data['Neighborhood'].apply(encode)

def encode_1(x):
     return 1 if x == 'PosA' or x == 'PosN' or x == 'RRNn' else 0#feat highest price return 1
train_data['Condition1']=train_data['Condition1'].apply(encode)

def encode_1(x):
     return 1 if x == 'PosA' or x == 'PosN' else 0#feat highest price return 1
train_data['Condition2']=train_data['Condition2'].apply(encode)

def encode_1(x):
     return 1 if x == '1Fam' or x == 'TwnhsE' else 0#feat highest price return 1
train_data['BldgType']=train_data['BldgType'].apply(encode)


def encode_1(x):
     return 1 if x == '2.5Fin' or x == '2Story' else 0#feat highest price return 1
train_data['HouseStyle']=train_data['HouseStyle'].apply(encode)


data.drop("RoofStyle", axis=1, inplace=True)

def encode_1(x):
     return 1 if x == 'WdShngl' else 0#feat highest price return 1
train_data['RoofMatl']=train_data['RoofMatl'].apply(encode)

def encode_1(x):
     return 1 if x == 'CemntBd' or x == 'ImStucc' or x == 'Stone' else 0#feat highest price return 1
train_data['Exterior1st']=train_data['Exterior1st'].apply(encode)

def encode_1(x):
     return 1 if x == 'Other' else 0#feat highest price return 1
train_data['Exterior2nd']=train_data['Exterior2nd'].apply(encode)


"""plt.scatter(train_data['GarageArea'], train_data['SalePrice'])
plt.xlabel('living area square feet')
plt.ylabel('sale price')
plt.show()"""

print("--- %s seconds ---" % (time.time() - start_time))

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import poisson, ols
from statsmodels.formula.api import negativebinomial
import numpy as np
import matplotlib.pyplot as plt
import math
# reading data
data_crime = pd.read_excel(r'C:\Users\digiovanniyani\Desktop\NY_CRIMES.xlsx') #, names = ['MH_MURD', 'MH_RAPE', 'MH_ROBB', 'MH_ASSA', 'MH_BURG', 'MH_LARC'])#, delim_whitespace=True, header=0)

total_nyc_crime=data_crime["MH_TOT"]+ data_crime["BK_TOT"]+ data_crime["QN_TOT"] + data_crime["BX_TOT"] + data_crime["SI_TOT"]

sumcols_brooklyn= data_crime["BK_LARC"]+ data_crime["BK_ROBB"]+ data_crime["BK_MOTO"] + data_crime["BK_BURG"] + data_crime["BK_ASSA"]
sumcols_queens= data_crime["QN_LARC"]+ data_crime["QN_ROBB"]+ data_crime["QN_MOTO"] + data_crime["QN_BURG"] + data_crime["QN_ASSA"]
sumcols_manhattan= data_crime["MH_LARC"]+ data_crime["MH_ROBB"]+ data_crime["MH_MOTO"] + data_crime["MH_BURG"] #+ data_crime["MH_ASSA"]
sumcols_staten= data_crime["SI_LARC"]+ data_crime["SI_ROBB"]+ data_crime["SI_MOTO"] + data_crime["SI_BURG"] + data_crime["SI_ASSA"]
import random


dates=(list(range(2005,2020)))
str_dates = [str(i) for i in dates]
#+ BK_HSGR + BK_UNEM
m1 = ols('QN_TOT ~ QN_HSGR + QN_INC + QN_UNEM  ', data = data_crime).fit() #maxiter=1000, method='nm')
print (m1.summary())
Y=data_crime['QN_TOT']
print(Y)
preds = m1.predict()

plt.plot(range(len(Y)), Y, 'r*-', range(len(Y)), preds, 'bo-')

plt.title('NYC total crimes VS  queens grad rate, unemployment,income')
plt.xticks(np.arange(0,15 , step=1))  # Set label locations.
plt.xticks(np.arange(0,15), [i for i in str_dates], rotation=30)
plt.show()


"""
data_row=data.iloc[2, 1:5]
data_row2=data.iloc[3, 1:5]
precip_list=data['PRECIP'].tolist()
print(type(data['PRECIP']))
print(type(data_row))


bb_count_list=data['BB_COUNT'].tolist()
data_row_string=data_row.to_string()
data_row2_string=data_row2.to_string()

#print(data_row2_string, 'd.d.d.d.d')
varcomp = 'data_row ~ data_row2'

m1 = poisson(varcomp, data).fit()
#m2=scipy.stats.poisson(data_row,data_row2 )
print( m1.summary())


#print("==========",data_row)


model_fit1 = data
preds_1 = m1.predict()
model_fit1['preds'] = preds_1

list_hight=data['PRECIP'].tolist()

#print(model_fit1.head(len(list_hight)))

list_preds=data['preds'].tolist()
list_bike_count=data['BB_COUNT'].tolist()
#print(list_preds)

x=np.arange(10)
plt.xticks(x,data['Date'])

y=np.arange(len(list_bike_count[10::5]))
plt.plot(y, list_bike_count[10::5],'-o')
plt.plot(y, list_preds[10::5], '-o')

plt.show()
"""
"""
params = [np.random.rand(), np.random.rand()]

new_params = scipy.optimize.minimize(likelihood, params, args=(data['HIGH_T'], data['BB_COUNT']))

a, b = new_params.x

# Plot
plt.plot(data['HIGH_T'], data['BB_COUNT'], 'o')
plt.plot(data['HIGH_T'], np.exp(a + b*data['HIGH_T']))
plt.text(0, 0, "a={:8.3f}, b={:8.3f}".format(a, b))
print(a,b)
plt.show()
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn import datasets, linear_model, metrics
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import xlrd
from xlrd import open_workbook
import math

dates=(list(range(2005,2020)))
str_dates = [str(i) for i in dates]

data_crime = pd.read_excel(r'C:\Users\digiovanniyani\Desktop\NY_CRIMES.xlsx')
#+ QN_UNEM + QN_INC
sumcols_brooklyn= data_crime["BK_LARC"]+ data_crime["BK_ROBB"]+ data_crime["BK_MOTO"] + data_crime["BK_BURG"] + data_crime["BK_ASSA"]
sumcols_manhattan= data_crime["MH_LARC"]+ data_crime["MH_ROBB"]+ data_crime["MH_MOTO"] + data_crime["MH_BURG"] + data_crime["MH_ASSA"]
sumcols_manhattan_2=data_crime["MH_TOT"]-sumcols_manhattan
m1 = smf.ols('BX_TOT ~ MH_HSGR + MH_INC + MH_UNEM ', data = data_crime).fit()#maxiter=1000, method='nm')

#pearson_coef, p_value = stats.pearsonr(data_crime['QN_UNEM'], data_crime['MH_TOT'])
#print(pearson_coef,"CORR COEFF SIMPLE")
#print(math.sqrt(m1.rsquared), "MULTIPLE CORR COEFF")
print (m1.summary())

print(m1.f_pvalue)

Y= data_crime['BX_TOT']
#print(Y, '===actual crimes')
preds = m1.predict()
print(preds, '======++++++======predicted')
X = np.arange(len(Y))
#print(m1.rsquared)
plt.plot(range(len(Y)), Y, 'r*-',label='actual')
plt.plot( range(len(Y)), preds, 'bo-', label='predicted')
plt.xticks(np.arange(0,15), [i for i in str_dates], rotation=30)
plt.xlabel('BX_HSGR + BX_POV + BX_UNEM', fontweight='bold', color = 'red', fontsize='12', horizontalalignment='center')
plt.ylabel("MH_TOT", fontweight='bold', color = 'red', fontsize='12', horizontalalignment='center')
plt.legend()
plt.tight_layout()
plt.show()

tot_crime= data_crime['MH_TOT'] + data_crime['BK_TOT'] +data_crime['BX_TOT'] +data_crime['QN_TOT'] +data_crime['QN_TOT']
print(tot_crime)
plt.xticks(np.arange(0,15), [i for i in str_dates], rotation=30)
plt.plot(tot_crime)
plt.ylabel("New York City Land Area Total Crime", fontweight='bold', color = 'red', fontsize='12', horizontalalignment='center')
plt.tight_layout()
plt.show()

plt.title('SI_TOT vs SI_HSGR-UNEM-INC')
plt.bar( X + 0.00,Y, color = 'b', width = 0.25, label='actual') #X + 0.00,
plt.bar(X+0.25 + 0.00, preds, color = 'g', width = 0.25, label='pred') #X + 0.25,
plt.xticks(np.arange(0,15 , step=1))  # Set label locations.
plt.xticks(np.arange(0,15), [i for i in str_dates], rotation=30)
plt.legend()
plt.show()

plt.scatter(data_crime['QN_UNEM'],data_crime['MH_TOT'])
#plt.plot(range(len(Y)), Y, 'r*-', label='actual')
#plt.plot (range(len(Y)), preds, 'bo-',label='pred')
plt.show()

plt.title('SI_TOT vs SI_HSGR-UNEM-INC')
plt.xticks(np.arange(0,15 , step=1))  # Set label locations.
plt.xticks(np.arange(0,15), [i for i in str_dates], rotation=30)
plt.legend()
plt.show()
indep_var='BK_TOT'
dep_var='MH_TOT'
xx = np.array(data_crime[indep_var])
yy = np.array(data_crime[dep_var])
plt.plot(xx, yy, 'o')
pearson_coef, p_value = stats.pearsonr( data_crime[dep_var],data_crime[indep_var])
print(pearson_coef,"CORR COEFF SIMPLE")
  # Set label locations.
#plt.xticks( np.arange(0,15, step=1),xx)
plt.xlabel("Bronx avg househom")
#plt.xticks(np.arange(0,15 , ld income (thousands)", fontweight='bold', color = 'red', fontsize='10', horizontalalignment='center')
plt.ylabel("Manhattan total crime", fontweight='bold', color = 'red', fontsize='10', horizontalalignment='center')





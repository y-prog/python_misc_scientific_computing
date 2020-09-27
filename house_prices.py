import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import xlrd

#C:\Users\digiovanniyani\Downloads


train_data=pd.read_csv(r'C:\Users\digiovanniyani\Downloads\train.csv')

#print(train_data.head())

#sale_price=train_data['SalePrice']
#plt.hist(np.log(sale_price), 25)
#plt.show()
#print(train_data.shape)

train_data_list=  train_data.values.tolist()
#train_data_list.append(train_data)   #  =train_data.to_numpy()
list1=list(train_data.shape)
rows=list1[0]
cols=list1[1]

#list_split=[train_data_list[i:i+b] for i  in range(0, len(train_data_list), b)]

#print(train_data_list[0][1:3])

#print((train_data_list[0][0:b]))
my_array=np.array(train_data_list)
shape=(rows,cols)

my_array.reshape(shape)


def column(matrix, i):
    return [row[i] for row in matrix]

print (type(column(my_array,2)[1]))
print(type(column(my_array,6)[1]))


def file_values(x):
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











#null_values='nan'
print((file_values()))




"""if column(my_array, i)[j] == np.nan:
my_counter +=1
print(my_counter)"""







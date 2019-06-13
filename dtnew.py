
from astral import Astral
import numpy as np
import matplotlib.pyplot as plt
import datetime 
import pytz
import sys

"""
TASK1--- open txt file, fix datas alignment and data corruption
"""


def last_digits(num, last_digits_count=2):
    return abs(num) % (10**last_digits_count)
def lastdig(num, lastdigcount=1):
    return abs(num)%(10**lastdigcount)

                
dt_fmt = '%Y-%m-%d %H:%M:%S.%f'

orig_date=[]
orig_time=[]
movements=[]

with open('moredates.txt', 'r') as f:
    for line in f:
        data = line.split()    # Splits on whitespace        
        orig_date.append(data[0][:])
        orig_time.append((data[1][:]))
        movements.append(int(data[2][:]))
        
    for i in range(1, len(orig_date)-1):
        if (len(str(movements[i])) <= 3 ):
            if ((movements[i]==0 and movements[i+1] != 0)  
            or (  (movements[i+1] == movements[i-1])  and 
             (movements[i] != movements[i-1]) and 
             (last_digits(movements[i]) == last_digits(movements[i-1])))):
            
                new_val = round((movements[i-1]+movements[i+1])/2)
                print('Index of changed element =',i, '\nPrevious value =',movements[i], '\nNew value =',new_val, '\n')
                movements[i] = new_val


    for j in range(0,len(orig_date)):
        if abs(movements[j-1]-movements[j])>8:
            if (movements[j-1]-movements[j] >0):
                nval=movements[j] + 8
                print('Index of changed element =',j-1, '\nPrevious value =',movements[j-1], '\nNew value =',nval, '\n')
                nval=movements[j-1]
            else:
                nval1=movements[j-1]+8
                print('Index of changed element =',j, '\nPrevious value =',movements[j], '\nNew value =',nval1, '\n')
                nval1=movements[j]

""" TASK2--- Convert from UCT to CEU---------"""
timestamps = []

for col_dt in zip(orig_date , orig_time):
    
    new_dt_str = ' '.join(col_dt)
    new_dt = datetime.datetime.strptime(new_dt_str, dt_fmt)
    timestamps.append(new_dt)
    

def convert_local_timezone():
    """ conversion from strings to datetime objects"""
    converted_dates=[]
    for date in timestamps:
        local_tz = pytz.timezone('Europe/Copenhagen')
        local_time = date.replace(tzinfo=pytz.utc).astimezone(local_tz)
        converted_dates.append(local_time)
    return converted_dates

CEU_times=convert_local_timezone()

"""TASK3---- Calculate movements per hour, show in bar graph"""
def mov_index(i,j):
    a=[movements[i] - movements[i-1] for i in range(i,j,30)]  # function calculating
    return a                                                  # the values                          

position1 =  orig_date.index(('2015-05-12'))+30
position2 = orig_date.index('2015-05-13')+30
position3 = orig_date.index('2015-05-14')+30


start_date = CEU_times[position1].date()

end_date=CEU_times[position3].date()
y=[mov_index(position1,position3-30)][0][:]
 
date_list = [start_date + datetime.timedelta(days=x) for x in range(0, 3)] 


city_name = 'Copenhagen'
a = Astral()                  #needed to find out if movements take place during daytime/nighttime
a.solar_depression = 'civil'
city = a[city_name]

sun1=city.sun(date=(CEU_times[position1]), local=True)
sun2=city.sun(date=(CEU_times[position2]), local = True)
sun3=city.sun(date=(CEU_times[position3]), local=True)


sun1_dawn_float=int(str(sun1['dawn'].time())[0:2]) + round((int(str(sun1['dawn'].time())[3:5])/60),2) -2
sun1_sunset_float=int(str(sun1['sunset'].time())[0:2]) + round((int(str(sun1['sunset'].time())[3:5])/60) ,2) -2
sun2_dawn_float=int(str(sun2['dawn'].time())[0:2]) + round((int(str(sun2['dawn'].time())[3:5])/60),2) +24 -2
sun2_sunset_float=int(str(sun2['sunset'].time())[0:2]) + round((int(str(sun2['sunset'].time())[3:5])/60),2) +24 -2
sun3_dawn_float=int(str(sun3['dawn'].time())[0:2]) + round((int(str(sun3['dawn'].time())[3:5])/60),2) +24*2 -2
sun3_sunset_float=int(str(sun3['sunset'].time())[0:2]) + round((int(str(sun3['sunset'].time())[3:5])/60) ,2)+24*2 -2


time_list=[CEU_times[position1] + datetime.timedelta(hours=i) for i in range(0,len(y))]
print(CEU_times[position1:position1+4])

hours=[time_list[i].time() for i in range(0,len(y),3)]

plt.axvspan(sun1_dawn_float,sun1_sunset_float,facecolor='green', alpha=3)
plt.axvspan(sun1_sunset_float,sun2_dawn_float, facecolor='orange',alpha=3)
plt.axvspan(sun2_dawn_float,sun2_sunset_float, facecolor='green',alpha=3)

plt.bar(np.arange(len(y[0:-1])),y[0:-1], width=1, color='blue' )

plt.bar(np.arange(len(y)),y, width=1, color='blue' )

plt.xticks(range(1, len(y),3 ),hours, rotation=45)

plt.gcf().autofmt_xdate()
plt.twiny()
plt.xticks((1,23,45), date_list, rotation=45)

legend_elements = [Line2D([0], [0], color='green', lw=4, label='daytime'),
                   Line2D([0], [0], color='orange', label='nighttime'),
                   Line2D([0], [0], color='blue', label='movements per hour')]

plt.legend(handles=legend_elements, loc='upper right')

plt.show()

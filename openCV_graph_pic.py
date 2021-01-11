



#obtaining numerical information from graph

import cv2
import numpy as np

file_name= (r"C:\Users\digiovanniyani\Desktop\nyc_income.png")

import cv2
import numpy as np

img = cv2.imread(file_name)
list_size=list( img.shape)
print(list_size)

number_of_x_values=13
number_of_y_values=70000
starting_x_value=2004
ending_y_value=90000
x_axis=list_size[1]
y_axis=list_size[0]



def click_event(event, x, y, flags, param):
    xx=(number_of_x_values/x_axis)*x + starting_x_value
    yy= -y*(number_of_y_values/y_axis)+ ending_y_value
    a=np.array(xx)
    b=np.array(yy)

    if event == cv2.EVENT_LBUTTONDOWN:
        print(a,b)

par = [(200,0),(200,0,0)]

cv2.imshow('original', img)
cv2.setMouseCallback("original", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows
#try function

import time
#import sklearn as scikit
from sklearn.linear_model import LinearRegression
import numpy as numpy
import matplotlib.pylab as plt
import math

booking_id = 0
lead_time = 1
arrival_year = 2
arrival_month = 3
arrival_date = 4
num_weekend_nights = 5
num_week_nights = 6
meal_plan = 7
parking = 8
room_type = 9
num_adults = 10
num_children = 11
market_segment = 12
repeated_guest = 13
num_prev_cancellations = 14
num_previous_non_cancelled = 15
avg_room_price = 16
special_requests = 17
booking_status = 18

def get_data(filename):

    data = open(filename, "r")
    lines = data.readlines()
    lines.pop(0)

    DATA = []


    for i in range(18):
        DATA.append([])

    for line in lines:
        line_data = line.split(",")
        for i in range(len(DATA)):
            DATA[i].append(line_data[i])

    return DATA

def get_narray(array):
    x = []
    for i in range(len(array)-1):
        arr = [array[i], array[i+1]]
        for j in range(len(arr)):
            arr[j] = int(math.floor(float(arr[j])))
        x.append(arr)

    return numpy.array(x)


DATA = get_data("trainingdata.csv")

X = get_narray(DATA[num_adults])
Y = get_narray(DATA[avg_room_price])


mod = LinearRegression().fit(X,Y)
#pred = mod.predict(X)

#plt.scatter(pred,Y)
plt.scatter(X,Y)
plt.show()
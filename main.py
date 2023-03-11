import time
import sklearn as scikit

data = open("trainingdata.csv", "r")
lines = data.readlines()
lines.pop(0)

DATA = []

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

for i in range(18):
    DATA.append([])

for line in lines:
    line_data = line.split(",")
    for i in range(len(DATA)):
        DATA[i].append(line_data[i])

print(DATA[arrival_date][0])

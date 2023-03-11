import time
import pandas as pd
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
sns.set_palette(sns.color_palette(['#851836', '#edbd17']))
sns.set_style("darkgrid")

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Function for replacing string data into numeric values
def get_numeric_value(datf, items):
    for i in range(len(items)):
        datf = datf.replace({items[i]: i})
    return datf


# Get the Dataframe
df = pd.read_csv('trainingdata.csv')

# Replace all string data with numeric values
df['MealPlan'] = get_numeric_value(df['MealPlan'], ['Not Selected', 'Meal Plan 1', 'Meal Plan 2'])

df['RoomType'] = get_numeric_value(df['RoomType'], ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6'])

df['MarketSegment'] = get_numeric_value(df['MarketSegment'], ['Offline', 'Online', 'Corporate', 'Complementary'])

df['BookingStatus'] = get_numeric_value(df['BookingStatus'], ['Canceled', 'Not_Canceled'])

#Remove parking, arrival year, repeated guest (PREDICT BOOKING STATUS)
df = df.drop(columns=['Parking', 'ArrivalYear', 'RepeatedGuest'])

#Add number of children and adults
#to make number of people column
i = 0
for item in df['NumAdults']:
    df['NumAdults'][i] = df['NumAdults'][i] + df['NumChildren'][i]
    i+=1
df = df.rename(columns={"NumAdults":"NumPeople"})

#Change NumChildren to HasChildren
i = 0
for item in df['NumChildren']:
    num = df['NumChildren'][i]

    if num > 0:
        df['NumChildren'][i] = 1
    else:
        df['NumChildren'][i] = 0
    
    i+=1
df = df.rename(columns={'NumChildren':'HasChildren'})

# df[['AvgPriceMean', 'AdultCount','ChildCount','AvgNumWeekNights', 'AvgNumWeekendNights']] = grouped_df[['AvgRoomPrice', 'NumAdults','NumChildren', 'NumWeekNights', 'NumWeekendNights']].transform('mean')
# df[['Canceled']] = grouped_df[['BookingStatus']].transform('sum')
# pd.set_option('display.max_columns', None)

# print (df[['RoomType', 'AvgPriceMean','AvgNumWeekNights', 'AvgNumWeekendNights', 'AdultCount', 'ChildCount', 'Canceled']].head(50))

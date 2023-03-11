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

#Get the Dataframe
df = pd.read_csv('trainingdata.csv')
print (df.head)

#Change the string values to numerical values
df['BookingStatus'] = df['BookingStatus'].replace({'Not_Canceled':0, 'Canceled': 1})
print (df.head)

df['RoomType'] = df['BookingStatus'].replace({'Room_Type 1': 1, 'Room_Type 2': 2,'Room_Type 3': 3,'Room_Type 4': 4,'Room_Type 5': 5,'Room_Type 6': 6})
print (df.head)

df['MealPlan'] = df['MealPlan'].replace({'Not Selected': 0, 'Meal Plan 1': 1, 'Meal Plan 2' : 2})
print (df.head)

df['MarketSegment'] = df['MarketSegment'].replace({'Offline' : 0,'Online' : 1,'Corporate' : 2,'Complementary' : 3})



#grouped_df = df.groupby('RoomType')

#df[['AvgPriceMean', 'AdultCount','ChildCount','AvgNumWeekNights', 'AvgNumWeekendNights']] = grouped_df[['AvgRoomPrice', 'NumAdults','NumChildren', 'NumWeekNights', 'NumWeekendNights']].transform('mean')
#df[['Canceled']] = grouped_df[['BookingStatus']].transform('sum')
#pd.set_option('display.max_columns', None)

#print (df[['RoomType', 'AvgPriceMean','AvgNumWeekNights', 'AvgNumWeekendNights', 'AdultCount', 'ChildCount', 'Canceled']].head(50))
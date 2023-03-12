import time
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

sns.set_theme()
sns.set_palette(sns.color_palette(['#851836', '#edbd17']))
sns.set_style("darkgrid")


# Function for replacing string data into numeric values
def get_numeric_value(datf):
    # Create list to store all different strings in the column
    items = []
    for i in range(len(datf)):
        if not datf[i] in items:
            # String is not in the list, add it at the end
            things.append(datf[i])

    for i in range(len(items)):
        # Replace string with index integer
        datf = datf.replace({items[i]: i})

    return datf


# Get the Dataframe
df = pd.read_csv('trainingdata.csv')
print(df)
# Replace all string data with numeric values
df['MealPlan'] = get_numeric_value(df['MealPlan'])

df['RoomType'] = get_numeric_value(df['RoomType'])

df['MarketSegment'] = get_numeric_value(df['MarketSegment'])

df['BookingStatus'] = get_numeric_value(df['BookingStatus'])

# Remove parking, arrival year, repeated guest (PREDICT BOOKING STATUS)
print(df)

df['Number of Guests'] = df[['NumAdults', 'NumChildren']].sum(axis=1)
df['Has Children'] = np.where(df['NumChildren'] > 0, 1, 0)

# df[['AvgPriceMean', 'AdultCount','ChildCount','AvgNumWeekNights', 'AvgNumWeekendNights']] = grouped_df[['AvgRoomPrice', 'NumAdults','NumChildren', 'NumWeekNights', 'NumWeekendNights']].transform('mean')
# df[['Canceled']] = grouped_df[['BookingStatus']].transform('sum')
# pd.set_option('display.max_columns', None)

# print (df[['RoomType', 'AvgPriceMean','AvgNumWeekNights', 'AvgNumWeekendNights', 'AdultCount', 'ChildCount', 'Canceled']].head(50))

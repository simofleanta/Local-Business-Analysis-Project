import pandas as pd
import seaborn as sns 
from pandas import DataFrame
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import numpy as np
import plotly
import statistics
import plotly.express as px
import stats
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import plotly.express as px

c=pd.read_csv('bike_business_plan.csv')
print(c.columns)
df=DataFrame(c.head(500))
print(df.head(500))

#pivots. 
pivot1=df.pivot_table(index='Season',columns='Item', aggfunc={'Number_Bikes':'count'}).fillna(0)
pivot1['Max']=pivot1.idxmax(axis=1)
print(pivot1)


#What are the min max levels of items and months for the business?

pivot2=df.pivot_table(index='Season',columns='Item', aggfunc={'Sales':'count'}).fillna(0)
pivot2['Max']=pivot2.idxmax(axis=1)
print(pivot2)

pivotday=df.pivot_table(index='Day',columns='Item', aggfunc={'Sales':'count'}).fillna(0)
pivotday['Max']=pivotday.idxmax(axis=1)
print(pivotday)

pivotday_m=df.pivot_table(index='Day',columns=['Year','Month','Item'], aggfunc={'Sales':'sum'}).fillna(0)
pivotday_m['Max']=pivotday_m.idxmax(axis=1)
print(pivotday_m)

pivotday_min=df.pivot_table(index='Month',columns=['Year','Item'], aggfunc={'Sales':'min'}).fillna(0)
pivotday_min['Min']=pivotday_min.idxmin(axis=1)
print(pivotday_min)

y19=df[df.Year==2019]
encoder=LabelEncoder()
df['Sales']=encoder.fit_transform(df['Sales'])
sns.violinplot(x=y19["Item"], y=y19["Sales"], palette="Blues")
plt.show()

df['Sales']=encoder.fit_transform(df['Sales'])
sns.violinplot(x=y19["Month"], y=y19["Sales"], palette="Blues")
plt.show()

#Pivot Bikes situ in 2020  

y=df[df.Year==2020]
pivotday_min_2020=y.pivot_table(index='Month',columns=['Year','Item'], aggfunc={'Sales':'min'}).fillna(0)
pivotday_min_2020['Min']=pivotday_min_2020.idxmin(axis=1)
print(pivotday_min_2020)

pivotday_max_2020=y.pivot_table(index='Month',columns=['Year','Month','Item'], aggfunc={'Sales':'max'}).fillna(0)
pivotday_max_2020['Max']=pivotday_max_2020.idxmax(axis=1)
print(pivotday_max_2020)

#Pivots to show day 2  max values for months

#pivot day2 2020
day2=y[y.Day==2]
pivotday2_2020=y.pivot_table(index='Item',columns=['Month','Item'], aggfunc={'Sales':'max'}).fillna(0)
pivotday2_2020['Max']=pivotday2_2020.idxmax(axis=1)
print(pivotday2_2020)

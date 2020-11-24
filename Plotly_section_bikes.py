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
#print(c.columns)
df=DataFrame(c.head(700))
print(df.head(700))

# Numerical
encoder=LabelEncoder()
df['Sales']=encoder.fit_transform(df['Sales'])
df['Number_Bikes']=encoder.fit_transform(df['Number_Bikes'])


c=df.select_dtypes(object)
#print(c)

c=df.dtypes
#print(c)

#groupings
x=df.groupby(['Season'])[['Number_Bikes']]
#print(x.mean())


fig1 = px.bar(df, x="Item", y=["Sales","Year"],barmode='group', color='Year',color_continuous_scale='Blues',title="comparing items in years 2020-2019")
plotly.offline.plot(fig1, filename='bike')

fig = px.bar(df, x="Item", y=["Sales","Year"],barmode='stack', color='Year',color_continuous_scale='Blues',title="Bike stacked per years")
plotly.offline.plot(fig, filename='bike')

#avg sales/mth

bike_d=df.groupby(['Month'])['Sales'].mean()
days=pd.DataFrame(data=bike_d)
bike_Month=days.sort_values(by='Sales',ascending=False,axis=0)
print(bike_Month)

fig = px.bar(bike_Month, x="Sales", y=bike_Month.index, color='Sales',color_continuous_scale='Blues',title="Average sales per month")
plotly.offline.plot(fig, filename='bike')

#avg sales month for a certain bike item in a year

y19=df[df.Year==2019]
Treck=y19[y19.Item=='Treck']

bike_d=Treck.groupby(['Month'])['Sales'].mean()
days=pd.DataFrame(data=bike_d)
bike_Year=days.sort_values(by='Sales',ascending=False,axis=0)

fig = px.bar(bike_Year, x="Sales", y=bike_Year.index, color='Sales',color_continuous_scale='Blues',title="Average  Treck sales per month in 2019")
plotly.offline.plot(fig, filename='bike')

#avg per month for year 2020 Treck

y20=df[df.Year==2020]
Amsterdam=y20[y20.Item=='Amsterdam']

bike_d=Amsterdam.groupby(['Month'])['Sales'].mean()
days=pd.DataFrame(data=bike_d)
bike_Year=days.sort_values(by='Sales',ascending=False,axis=0)

fig = px.bar(bike_Year, x="Sales", y=bike_Year.index, color='Sales',color_continuous_scale='Blues',title="Average  Amsterdam sales per month in 2020")
plotly.offline.plot(fig, filename='bike')

#avg bikes 
bike_d=df.groupby(['Item'])['Sales'].mean()
days=pd.DataFrame(data=bike_d)
bike_Item=days.sort_values(by='Sales',ascending=False,axis=0)

fig = px.bar(bike_Item, x="Sales", y=bike_Item.index, color='Sales',color_continuous_scale='Blues',title="Average sales per month")
plotly.offline.plot(fig, filename='bike')

import plotly.express as px
df = px.data.tips()
fig = px.density_heatmap(y20, x="Day", y="Sales", nbinsx=20, nbinsy=20, color_continuous_scale="Blues",title='sales distribution on all days in 2020')
plotly.offline.plot(fig, filename='bike')

Month=y20['Month']
Sales=y20['Sales']
Season=y20['Season']
Item=y20['Item']

fig = go.Figure(data=go.Heatmap(                   
                   x=Item,
                   y=Season,
                   z=Sales,
                   colorscale='RdBu'))

fig.update_layout(
    title='Bikes items by season sales 2020',
    xaxis_nticks=36)


Month=y19['Month']
Sales=y20['Sales']
Season=y20['Season']
Item=y19['Item']
daytime=y20['Day_Time']

fig = go.Figure(data=go.Heatmap(                   
                   x=Season,
                   y=daytime,
                   z=Sales,
                   colorscale='RdBu'))

fig.update_layout(
    title='Bikes items by month sales 2019',
    xaxis_nticks=36)


plotly.offline.plot(fig, filename='bike')





















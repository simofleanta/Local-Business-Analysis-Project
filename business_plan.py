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

"""A friend has a bike business and wants to see the business evolution given the pandemic situ
if 2019 is better than 2020
he would like to see what bikes sell best?
what are the best months and days?"""

#EDA-gropings, sortings, mean, max, sum values in aggregs 
#pivotations
#visuals with seaborn
#visuals with plotly (also a separate section containing plotly)
#function on roi 
#a/b approach 
#profitability 



c=pd.read_csv('bike_business_plan.csv')
print(c.columns)
df=DataFrame(c.head(500))
print(df.head(500))

#ncoe to numeric
encoder=LabelEncoder()
df['Sales']=encoder.fit_transform(df['Sales'])

sns.violinplot(x=df["Item"], y=df["Sales"], palette="Blues")
plt.show()

""" making sense of data"""
c=df.select_dtypes(object)
#print(c)

#trasnform in numerical
encoder=LabelEncoder()
df['Number_Bikes']=encoder.fit_transform(df['Number_Bikes'])

c=df.dtypes
#print(c)

"""Exploratory data"""
#groupings
x=df.groupby(['Season'])[['Number_Bikes']]
print(x.mean())

#Aggregate
operations=['mean','sum','min','max']
a=df.groupby(['Year','Month'], as_index=False)[['Number_Bikes']].agg(operations)
print(a.reset_index())

#sorting values
df['Number_Bikes'].value_counts().sort_values(ascending=False).head(10)
sns.violinplot(x=df["Month"], y=df["Number_Bikes"], palette="Blues")
plt.show()

#when is the bike business doing the tbest  during the day-time? 

fig, ax=plt.subplots(figsize=(6,4))
sns.set_style('darkgrid')
df.groupby('Day_Time')['Item'].count().sort_values().plot(kind='bar')
plt.ylabel('Number_Bikes')
ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.title('Business during the day')
plt.show()

#sort by day
sortbyday=df.groupby('Day_Time')['Item'].count().sort_values(ascending=False)

# what is business performance in the past months?

df.groupby('Item')['Month'].count().plot(kind='bar')
plt.ylabel('Number_Bikess')
plt.title('Bikes number during the past months')
plt.show()

#What is the situ in Oktober?
Okt=df.loc[df['Month']=='Okt'].nunique()

""" groupings """

#pivots. I should add the bike brand name so I can see which one is the pivot one
pivot1=df.pivot_table(index='Season',columns='Item', aggfunc={'Number_Bikes':'count'}).fillna(0)
pivot1['Max']=pivot1.idxmax(axis=1)
print(pivot1)

df.groupby('Month')['Sales'].sum().plot(kind='bar')
plt.ylabel('Sales')
plt.title('Bikes sales in the last months')
plt.show()

#Is 2019 better than 2020 given the pandemy?
df.groupby('Year')['Sales'].sum().plot(kind='bar')
plt.ylabel('Sales')
plt.title('2019-2020 comparison')
plt.show()

"""pivotations"""

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

#filter years
y19=df[df.Year==2019]

df['Sales']=encoder.fit_transform(df['Sales'])
sns.violinplot(x=y19["Item"], y=y19["Sales"], palette="Blues")
plt.show()

df['Sales']=encoder.fit_transform(df['Sales'])
sns.violinplot(x=y19["Month"], y=y19["Sales"], palette="Blues")
plt.show()

#Bikes situ in 2020  

y=df[df.Year==2020]
pivotday_min_2020=y.pivot_table(index='Month',columns=['Year','Item'], aggfunc={'Sales':'min'}).fillna(0)
pivotday_min_2020['Min']=pivotday_min_2020.idxmin(axis=1)
print(pivotday_min_2020)

pivotday_max_2020=y.pivot_table(index='Month',columns=['Year','Month','Item'], aggfunc={'Sales':'max'}).fillna(0)
pivotday_max_2020['Max']=pivotday_max_2020.idxmax(axis=1)
print(pivotday_max_2020)

#show me day 2  max values for months

#pivot day2 
day2=y[y.Day==2]
pivotday2_2020=y.pivot_table(index='Item',columns=['Month','Item'], aggfunc={'Sales':'max'}).fillna(0)
pivotday2_2020['Max']=pivotday2_2020.idxmax(axis=1)
print(pivotday2_2020)

#encode sales to numeric for violinplot
encoder=LabelEncoder()
df['Sales']=encoder.fit_transform(df['Sales'])
sns.violinplot(x=y["Item"], y=y["Sales"], palette="Blues")
plt.show()

df['Sales']=encoder.fit_transform(df['Sales'])
sns.violinplot(x=y["Month"], y=y["Sales"], palette="Blues")
plt.show()

# What are avg bikes sales in 2020

bike_d=y.groupby(['Item'])['Sales'].mean()
days=pd.DataFrame(data=bike_d)
bike_Item=days.sort_values(by='Sales',ascending=False,axis=0)

fig = px.bar(bike_Item, x="Sales", y=bike_Item.index, color='Sales',color_continuous_scale='Blues',title="Average sales per month")
plotly.offline.plot(fig, filename='bike')

#corr
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True,cmap='Blues_r',mask=np.triu(df.corr(),k=1))
plt.show()

"""ROI ON 2020 in a pandemic it was anticipated a larger use of echo transport including bikes 
instead of public transport so the investment was higher"""

investment=65000
bike_costs=2700
loss=5800

def roi(investment,bike_costs,loss):
    net_prof=bike_costs*12-loss
    roi=(net_prof/investment*100)
    return roi

ROI=roi(investment,bike_costs,loss)
print(ROI)

#on 2019
investment=40000
bike_costs=1000
loss=700

def roi(investment,bike_costs,loss):
    net_prof=bike_costs*12-loss
    roi=(net_prof/investment*100)
    return roi

ROI=roi(investment,bike_costs,loss)
print(ROI)

"""Given the differences between the hears, it worth using an A/B approach over ites in years or season/months """

a=df['Interested']
b=df['Likely']
c=df['Not_interested']
d=df['Not_likely']

#subset add calculation to dataset
#add df['a]=forumula
df['A']=df.Interested/df.Likely
df['B']=df.Not_interested/df.Not_likely
print(df.columns)

#print dataset with the situations A,B
print(df.head (3))

#aggregate ABs/season

Season_A=df.groupby(['Season','Item'])[['A']]
print(Season_A.mean())

Season_B=df.groupby(['Season','Item'])[['B']]
print(Season_B.mean())

#agg A/B /mth

Month_A=df.groupby(['Month','Item'])[['A']]
print(Month_A.mean())

Month_B=df.groupby(['Month','Item'])[['B']]
print(Month_B.mean())

#agg A/B per year #some items may not be found in certain years

Year_A=df.groupby(['Year','Item'])[['A']]
print(Year_A.mean())

Year_B=df.groupby(['Year','Item'])[['B']]
print(Year_B.mean())

#Graph on A situ
ab=df
df = px.data.tips()
fig = px.density_heatmap(ab, x="Item", y="A", nbinsx=20, nbinsy=20, color_continuous_scale="Blues",title='Situation A distribution occross items')




"""Calculate Profitability"""

#profitability of product forumula
#cost to produce =2000 per product *no of prods
#subtract cost to produce from revenues
#if profitability per product sold= product profitability / number of products 

df['Cost_to_produce']=2000*df.Number_Bikes
print(df.columns)
df['Profitability']=df.Cost_to_produce-df.Sales
df['Profitability_p']=df.Profitability/df.Number_Bikes
print(df.columns)
print(df.head(3))

"""Aggregate profitability per bike brand"""
Profitability_group=df.groupby(['Season','Item'])[['Profitability']]
print(Profitability_group.mean())

Profitability_p=df.groupby(['Season','Item'])[['Profitability_p']]
print(Profitability_p.mean())

df.groupby('Month')['Profitability'].sum().plot(kind='bar')
plt.ylabel('Profitability')
plt.title('Performance per month')
plt.show()

df.groupby('Item')['Profitability'].sum().plot(kind='bar')
plt.ylabel('Profitability')
plt.title('Item performance comparison')
plt.show()

df.groupby('Year')['Sales'].sum().plot(kind='bar')
plt.ylabel('Sales')
plt.title('2019-2020 comparison')
plt.show()
#

















































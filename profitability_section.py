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
#print(df.head(500))



#-----------------------------------PROFITABILITY--------------------------------
#due to dataset minor transforation, errors may occur for this section.Then please reffer to a separate 
#section profitability on the github where data is plain and no erros occur
#Calculate Profitability

#profitability of product forumula
#cost to produce =2000 per product *no of prods
#subtract cost to produce from revenues
#if profitability per product sold= product profitability / number of products 
#easy glance subsets

df['Cost_to_produce']=2000*df.Number_Bikes
df['Profitability']=df.Cost_to_produce-df.Sales
df['Profitability_p']=df.Profitability/df.Number_Bikes
print(df.columns)
print(df.head(3))

#Aggregate profitability per bike brand
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

autumn=df[df.Season=='autumn']
Month_sep=df[df.Month=='Sep']

p_stack=autumn.append(Month_sep)
profitab_s=p_stack[4:80][['Year','Item','Profitability','Sales']]
print(profitab_s.tail (5))





























































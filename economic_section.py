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

economic=pd.read_csv('unemployment.csv')
print(economic.columns)
df_economic=DataFrame(economic.head(10))
print(df_economic.head(10))

economic=pd.read_csv('Inflation_forecast.csv')
print(economic.columns)
df_inflation=DataFrame(economic.head(10))
print(df_inflation.head(10))


#merged the two datasets 
inflation_unemployment=pd.merge(df_inflation,df_economic)
print(inflation_unemployment)

#getting rid of mess in my table
#data taken from ins bnr
#data showing unemployment unregistered and supported 

Inflation_Unemployed_Table=inflation_unemployment[['Year','Months','Term','IPC','Constant_taxes','Annual_inflation_target','Number_unemployd']].copy()
print(Inflation_Unemployed_Table)


#corr graphs 
#Watching correlations, variations between unemployment and IPC. 
plt.figure(figsize=(8,5))
sns.heatmap(Inflation_Unemployed_Table.corr(),annot=True,cmap='Blues_r',mask=np.triu(Inflation_Unemployed_Table.corr(),k=1))
plt.show()

x=Inflation_Unemployed_Table['IPC']
y=Inflation_Unemployed_Table['Number_unemployd']
z=Inflation_Unemployed_Table['Term']
q=Inflation_Unemployed_Table['Annual_inflation_target']
p=Inflation_Unemployed_Table['Months']

#corr between ipc and number unemployed
fig = go.Figure(data=go.Heatmap(
                   z=x,
                   x=z,
                   y=y,
                   colorscale='Blues'))

fig.update_layout(
    
    title='Correlation IPC and number of unemployd people',
    xaxis_nticks=40)
plotly.offline.plot(fig, filename='eco')
#It seems that the larger IPC the greater number of unemployd people. This leads to the idea that 
#it is getting harder to get employeed. Due to lower puchase power, employers think twice before employeeng someone, 
#considering more compact jobs, qualified, digital type of jobs. 

#Correlation Inflation target and number of unemployd people in months
fig = go.Figure(data=go.Heatmap(
                   z=y,
                   x=p,
                   y=q,
                   colorscale='Blues'))

fig.update_layout(
    
    title='Correlation Inflation target and number of unemployd people in months',
    xaxis_nticks=40)
plotly.offline.plot(fig, filename='eco')

#bargraph on unemployd on monnths in 2020
Inflation_Unemployed_Table.groupby('Months')['Number_unemployd'].sum().plot(kind='bar')
plt.ylabel('Number_unemployd')
plt.title('Months comparison on number of unemployed')
plt.show()





















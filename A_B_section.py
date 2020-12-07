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
df=DataFrame(c.head(500))
#print(df.head(500))

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





































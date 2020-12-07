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


"""ROI ON 2020 in a pandemic it was anticipated a larger use of echo transport including bikes 
instead of public transport so the investment was higher"""

#2019 ROI
#filtering the year
Year2019=df[df.Year==2019]
investment=40000 #received investment 
#passing vriables to the desired columns 
bike_costs=Item_cost_month=Year2019['Item_cost_month']
loss=Loss_item=Year2019['Loss_item']

#finding out the netprofit
net_profit=bike_costs*12-loss

def ROI_2019(investment,bike_costs,loss):
    """function generating ROI for 2019"""
    return net_profit/investment*100
print(ROI_2019(investment,bike_costs,loss))

#roi/item-Raleigh 2019
#filtering the desired item in 2019
Raleigh=Year2019[Year2019.Item=='Raleigh']
investment=40000 #received investment 
bike_costs=Item_cost_month=Raleigh['Item_cost_month']
loss=Loss_item=Raleigh['Loss_item']

#finding out the netprofit 
net_profit=bike_costs*12-loss

def ROI_Ral(investment,bike_costs,loss):
    """Generating ROI for an item """
    return net_profit/investment*100
print(ROI_Ral(investment,bike_costs,loss))

#roi in 2020 
Year2020=df[df.Year==2020]
investment=40000 #received investment 
bike_costs=Item_cost_month=Year2020['Item_cost_month']
loss=Loss_item=Year2020['Loss_item']

net_profit=bike_costs*12-loss
def ROI_2020(investment,bike_costs,loss):
    return net_profit/investment*100
print(ROI_2020(investment,bike_costs,loss))

#roi item 2020
Year2020=df[df.Year==2020]
Orbea=Year2020[Year2020.Item=='Orbea']

investment=40000 #received investment 
bike_costs=Item_cost_month=Orbea['Item_cost_month']
loss=Loss_item=Orbea['Loss_item']

net_profit=bike_costs*12-loss
def ROI_Orbea(investment,bike_costs,loss):
    return net_profit/investment*100
print(ROI_Orbea(investment,bike_costs,loss))









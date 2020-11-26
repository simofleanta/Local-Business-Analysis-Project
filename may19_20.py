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

year_2019=df[df.Year==2019]
year_2020=df[df.Year==2020]

M=year_2019[df.Month=='May']
M1=year_2020[df.Month=='May']

stacked_ms=M.append(M1)
Months_stack=stacked_ms[4:20][['Year','Item','Month','Sales']]
print(Months_stack)


Months_stack.groupby(['Year','Item'])['Sales'].sum().plot(kind='bar')
plt.ylabel('Sales')
plt.title('2019-2020 comparison')
plt.show()






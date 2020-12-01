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

#pening a file
git=pd.read_csv('GitHub_data.csv')
#print(git.columns)
df=DataFrame(git.head(1000))
#print(df.head(20))

#data view
c=df.dtypes
c_missing=df.isnull().sum()
#print(c_missing)
x=c.describe()
#print(x)
x_describe=df.describe()
#print(x_describe)
x_shape=df.shape
#print(x_shape)

#just in case need to encode numerical 
#encoder=LabelEncoder()
#numerical=df['date']=encoder.fit_transform(df['date'])


#Making sense of data

xdf=df[['topic','projects','contributers','name','user','star','fork','issue','License','commits']].copy()
print(xdf)

#how many projects are per topic?
git_topic=xdf.groupby(['topic'])[['projects']]
print(git_topic.count())

#how many projects in total?
proj_count=xdf.groupby(['projects'])
print(proj_count.count())

#how many contribs per topic?
git_contribs=xdf.groupby(['topic'])[['contributers']]
print(git_contribs.count())

#how many contribs in projects
contribs_projects=xdf.groupby(['topic','projects'])[['contributers']]
print(contribs_projects.count())

#star is counted in k 
git_star=xdf.groupby(['projects'])[['star']]
print(git_star.mean())

git_name_star=xdf.groupby(['topic','name'])[['star']]
print(git_name_star.sum())

#forks are in k too 
git_license=xdf.groupby(['topic','License'])[['fork']]
print(git_license.sum())



















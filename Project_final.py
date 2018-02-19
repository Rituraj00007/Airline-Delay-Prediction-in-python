
# coding: utf-8

# In[1]:


### Data Loading 

import csv
import numpy as np
import pandas as pd
import time 
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('/user/ss186102/2008.csv')


# In[2]:


### Creation of Date field
df['date1'] = df.Year.astype(str).str.cat(df.Month.astype(str).str.zfill(2), sep='-').str.cat(df.DayOfWeek.astype(str).str.zfill(2), sep='-')

df['date1'] = df['date1'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))


# In[3]:


#Which carrier performs better
grouped= df['CarrierDelay'].groupby(df['UniqueCarrier']).mean()
#grouped.sort_values(['CarrierDelay'], ascending=True).plot(kind="bar")
grouped.plot(kind="bar")
print(grouped[grouped.min()==grouped])


# In[4]:


## When is the best day to minimise delay
Month_delays = df[['DepDelay', 'Month']].groupby('Month').mean()
Month_delays.plot(kind="bar")
di = {1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thursday',5:'Friday',6:'Saturday',7:'Sunday'}
df['dow']= df['DayOfWeek'].map(di)
DayOfWeek_delays = df[['DepDelay', 'dow']].groupby('dow').mean()
#DayOfWeek_delays = df[['CarrierDelay', 'DayOfWeek']].groupby('DayOfWeek').mean()
DayOfWeek_delays.sort_values(by=['DepDelay'], ascending=True).plot(kind="bar")

df['hour'] = df['CRSDepTime'].map(lambda x: int(str(int(x)).zfill(4)[:2]))
hour_delays = df[['DepDelay', 'hour']].groupby('hour').mean()

# Plotting average delays by hour of day
hour_delays.sort_values(by=['DepDelay'], ascending=True).plot(kind='bar')


# In[13]:


# Calculation of age of Carrier 
group = df[['date1']].groupby(df['UniqueCarrier'])
carrier_duration = group.max()-group.min()


carrier_duration.rename(columns = {'date1':'Age_of_Carrier(Months)'}, inplace = True)
carrier_duration['Age_of_Carrier(Months)'] = carrier_duration['Age_of_Carrier(Months)'].map(lambda x: round(int(str(x).replace('days','').split()[0])/30))

# Carrier Delay 
grouped= df['CarrierDelay'].groupby(df['UniqueCarrier'])
mean=grouped.mean()

## Cancellation of flight due to Carrier only 
gr1= df[df['CancellationCode']=='A'].UniqueCarrier.value_counts()
gr1= round(gr1[:]/100)
gr1 = pd.DataFrame(gr1)
gr1.rename(columns = {'UniqueCarrier':'Cancellation'}, inplace = True)

## Merging of data sets 
df1 = pd.concat([mean, carrier_duration,gr1], axis=1)
df1.plot(kind = 'bar')


# In[37]:


## Delay Propogation 
df1= df[['date1','DepTime','CRSDepTime','ArrTime', 'CRSArrTime', 'UniqueCarrier','FlightNum','DepDelay','ArrDelay','Origin', 'Dest',]]

df2 =pd.merge(df1, df1, left_on =['date1','UniqueCarrier','FlightNum','Dest'], right_on = ['date1','UniqueCarrier','FlightNum','Origin'],how='inner')

import operator
df2.loc[operator.and_(df2.DepDelay_x > 15, df2.DepDelay_y > 15 )].head()


# In[38]:


### Devloping a Model 

df = df[['UniqueCarrier', 'Origin', 'Dest',
        'CRSDepTime', 'DepTime', 'DepDelay',
        'CRSArrTime', 'ArrTime', 'ArrDelay',
        'CRSElapsedTime', 'ActualElapsedTime','AirTime',]]

missing_df = df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['variable', 'missing values']
missing_df['filling factor (%)']=(df.shape[0]-missing_df['missing values'])/df.shape[0]*100
missing_df.sort_values('filling factor (%)').reset_index(drop = True)


# In[39]:


# 97% it was filled so dropped few
df.dropna(inplace = True)


# In[40]:


# function that extract statistical parameters from a grouby objet:
def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}
#_______________________________________________________________
# Creation of a dataframe with statitical infos on each airline:
global_stats = df['DepDelay'].groupby(df['UniqueCarrier']).apply(get_stats).unstack()
global_stats = global_stats.sort_values('count')
global_stats


# In[41]:


import matplotlib.pyplot as plt
import seaborn as sns
delay_type = lambda x:((0,1)[x > 15],2)[x > 60]


df['DELAY_LEVEL'] = df['DepDelay'].apply(delay_type)

fig = plt.figure(1, figsize=(10,7))
ax = sns.countplot(y="UniqueCarrier", hue='DELAY_LEVEL', data=df)
# We replace the abbreviations by the full names of the companies and set the labels
labels = ax.get_yticklabels()
ax.set_yticklabels(labels)
plt.setp(ax.get_xticklabels(), fontsize=12, weight = 'normal', rotation = 0);
plt.setp(ax.get_yticklabels(), fontsize=12, weight = 'bold', rotation = 0);
ax.yaxis.label.set_visible(False)
plt.xlabel('Flight count', fontsize=16, weight = 'bold', labelpad=10)
#________________
# Set the legend
L = plt.legend()
L.get_texts()[0].set_text('on time (t < 15 min)')
L.get_texts()[1].set_text('small delay (15 < t < 60 min)')
L.get_texts()[2].set_text('large delay (t > 60 min)')
plt.show()


# In[43]:


# Import 
from sklearn.preprocessing import LabelEncoder
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, roc_curve
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().magic('matplotlib inline')
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rc('xtick', labelsize=8) 
matplotlib.rc('ytick', labelsize=8)
np.random.seed(10)
df = pd.read_csv('/user/ss186102/2008.csv')


# In[51]:



df1= df[['DayofMonth', 'DayOfWeek', 'CRSDepTime',
       'CRSArrTime', 'CRSElapsedTime',
        'AirTime', 'Distance',
       'ArrDelay']]
missing_df = df1.isnull().sum(axis=0).reset_index()
missing_df.columns = ['variable', 'missing values']
missing_df['filling factor (%)']=(df1.shape[0]-missing_df['missing values'])/df1.shape[0]*100
missing_df.sort_values('filling factor (%)').reset_index(drop = True)
df1.dropna(inplace = True)


# In[59]:


df1['Arrival_delay_classifier']= df1['ArrDelay'].map(lambda x: 0 if x>15 else 1)

df1.drop('ArrDelay',axis = 1,inplace=True)


# In[62]:


from sklearn.model_selection import train_test_split
train, valid = train_test_split(df1, test_size = 0.3,random_state=1991)
train=pd.DataFrame(train,columns=df1.columns)
valid=pd.DataFrame(valid,columns=df1.columns)
X_train=train.drop(['Arrival_delay_classifier'],axis=1)
Y_train=train['Arrival_delay_classifier']
X_valid=valid.drop(['Arrival_delay_classifier'],axis=1)
Y_valid=valid['Arrival_delay_classifier']


# In[67]:


from sklearn.metrics import roc_curve, auc  
def Performance(Model,Y,X):
    # Perforamnce of the model
    fpr, tpr, _ = roc_curve(Y, Model.predict_proba(X)[:,1])
    AUC  = auc(fpr, tpr)
    print ('the AUC is : %0.4f' %  AUC)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % AUC)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

#Performance(Model=RF0,Y=Y_valid,X=X_valid)


# In[72]:


RF0 = RandomForestClassifier(n_jobs = 1,random_state =1)
RF1= RF0.fit(X_train,Y_train)


# In[73]:


Performance(Model=RF1,Y=Y_valid,X=X_valid)


# In[77]:


sample_leaf_options = [1,10,20,75,100,200,500]
for leaf_size in sample_leaf_options :
    RF0 = RandomForestClassifier( n_jobs = -1,random_state =50,
                                   max_features = "auto", min_samples_leaf = leaf_size)
    RF1= RF0.fit(X_train,Y_train)
    Performance(Model=RF1,Y=Y_valid,X=X_valid)
         


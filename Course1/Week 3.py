
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# # Merging Dataframes
# 

# In[ ]:


import pandas as pd

df = pd.DataFrame([{'Name': 'Chris', 'Item Purchased': 'Sponge', 'Cost': 22.50},
                   {'Name': 'Kevyn', 'Item Purchased': 'Kitty Litter', 'Cost': 2.50},
                   {'Name': 'Filip', 'Item Purchased': 'Spoon', 'Cost': 5.00}],
                  index=['Store 1', 'Store 1', 'Store 2'])
df


# In[ ]:


df['Date'] = ['December 1', 'January 1', 'mid-May']
df


# In[ ]:


df['Delivered'] = True
df


# In[ ]:


df['Feedback'] = ['Positive', None, 'Negative']
df


# In[ ]:


adf = df.reset_index()
adf['Date'] = pd.Series({0: 'December 1', 2: 'mid-May'})
adf


# In[ ]:


staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR'},
                         {'Name': 'Sally', 'Role': 'Course liasion'},
                         {'Name': 'James', 'Role': 'Grader'}])
staff_df = staff_df.set_index('Name')
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business'},
                           {'Name': 'Mike', 'School': 'Law'},
                           {'Name': 'Sally', 'School': 'Engineering'}])
student_df = student_df.set_index('Name')
print(staff_df.head())
print()
print(student_df.head())


# In[ ]:


pd.merge(staff_df, student_df, how='outer', left_index=True, right_index=True)


# In[ ]:


pd.merge(staff_df, student_df, how='inner', left_index=True, right_index=True)


# In[ ]:


pd.merge(staff_df, student_df, how='left', left_index=True, right_index=True)


# In[ ]:


pd.merge(staff_df, student_df, how='right', left_index=True, right_index=True)


# In[ ]:


staff_df = staff_df.reset_index()
student_df = student_df.reset_index()
pd.merge(staff_df, student_df, how='left', left_on='Name', right_on='Name')


# In[ ]:


staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR', 'Location': 'State Street'},
                         {'Name': 'Sally', 'Role': 'Course liasion', 'Location': 'Washington Avenue'},
                         {'Name': 'James', 'Role': 'Grader', 'Location': 'Washington Avenue'}])
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business', 'Location': '1024 Billiard Avenue'},
                           {'Name': 'Mike', 'School': 'Law', 'Location': 'Fraternity House #22'},
                           {'Name': 'Sally', 'School': 'Engineering', 'Location': '512 Wilson Crescent'}])
pd.merge(staff_df, student_df, how='left', left_on='Name', right_on='Name')


# In[ ]:


staff_df = pd.DataFrame([{'First Name': 'Kelly', 'Last Name': 'Desjardins', 'Role': 'Director of HR'},
                         {'First Name': 'Sally', 'Last Name': 'Brooks', 'Role': 'Course liasion'},
                         {'First Name': 'James', 'Last Name': 'Wilde', 'Role': 'Grader'}])
student_df = pd.DataFrame([{'First Name': 'James', 'Last Name': 'Hammond', 'School': 'Business'},
                           {'First Name': 'Mike', 'Last Name': 'Smith', 'School': 'Law'},
                           {'First Name': 'Sally', 'Last Name': 'Brooks', 'School': 'Engineering'}])
staff_df
student_df
pd.merge(staff_df, student_df, how='inner', left_on=['First Name','Last Name'], right_on=['First Name','Last Name'])


# # Idiomatic Pandas: Making Code Pandorable

# In[1]:


import pandas as pd
df = pd.read_csv('census.csv')
df


# In[2]:


(df.where(df['SUMLEV']==50)
    .dropna()
    .set_index(['STNAME','CTYNAME'])
    .rename(columns={'ESTIMATESBASE2010': 'Estimates Base 2010'}))


# In[3]:


df = df[df['SUMLEV']==50]
df.set_index(['STNAME','CTYNAME'], inplace=True)
df.rename(columns={'ESTIMATESBASE2010': 'Estimates Base 2010'})


# In[4]:


import numpy as np
def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    return pd.Series({'min': np.min(data), 'max': np.max(data)})


# In[5]:


df.apply(min_max, axis=1)


# In[6]:


import numpy as np
def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    row['max'] = np.max(data)
    row['min'] = np.min(data)
    return row
df.apply(min_max, axis=1)


# In[7]:


rows = ['POPESTIMATE2010',
        'POPESTIMATE2011',
        'POPESTIMATE2012',
        'POPESTIMATE2013',
        'POPESTIMATE2014',
        'POPESTIMATE2015']
df.apply(lambda x: np.max(x[rows]), axis=1)


# # Group by

# In[ ]:


import pandas as pd
import numpy as np
idx = [1,2,3,1,2,3,4]
s = pd.Series([1,2,3,10,20,30,40], idx)
for group, ss in s.groupby(level=0):
    #print(group)
    print(ss)


# In[ ]:


import pandas as pd
import numpy as np
df = pd.read_csv('census.csv')
df = df[df['SUMLEV']==50]
df


# In[ ]:


get_ipython().run_cell_magic('timeit', '-n 10', "for state in df['STNAME'].unique():\n    avg = np.average(df.where(df['STNAME']==state).dropna()['CENSUS2010POP'])\n    print('Counties in state ' + state + ' have an average population of ' + str(avg))")


# In[ ]:


get_ipython().run_cell_magic('timeit', '-n 10', "for group, frame in df.groupby('STNAME'):\n    avg = np.average(frame['CENSUS2010POP'])\n    print('Counties in state ' + group + ' have an average population of ' + str(avg))")


# In[ ]:


df.head()


# In[ ]:


df = df.set_index('STNAME')

def fun(item):
    if item[0]<'M':
        return 0
    if item[0]<'Q':
        return 1
    return 2

for group, frame in df.groupby(fun):
    print('There are ' + str(len(frame)) + ' records in group ' + str(group) + ' for processing.')


# In[ ]:


df = pd.read_csv('census.csv')
df = df[df['SUMLEV']==50]


# In[ ]:


df.groupby('STNAME').agg({'CENSUS2010POP': np.average})


# In[ ]:


print(type(df.groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']))
print(type(df.groupby(level=0)['POPESTIMATE2010']))


# In[ ]:


(df.set_index('STNAME').groupby(level=0)['CENSUS2010POP']
    .agg({'avg': np.average, 'sum': np.sum}))


# In[ ]:


(df.set_index('STNAME').groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']
  .agg({'avg': np.average, 'sum': np.sum}))


# In[ ]:


(df.set_index('STNAME').groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']
    .agg({'POPESTIMATE2010': np.average, 'POPESTIMATE2011': np.sum}))


# # Scales

# In[ ]:


df = pd.DataFrame(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D'],
                  index=['excellent', 'excellent', 'excellent', 'good', 'good', 'good', 'ok', 'ok', 'ok', 'poor', 'poor'])
df.rename(columns={0: 'Grades'}, inplace=True)
df


# In[ ]:


df['Grades'].astype('category').head()
df['Grades'].astype('category')


# In[ ]:


grades = df['Grades'].astype('category',
                             categories=['E','D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+'],
                             ordered=True)
grades


# In[ ]:


grades > 'C'


# In[ ]:


s = pd.Series(['Low', 'Low', 'High', 'Medium', 'Low', 'High', 'Low'])
s.astype('category',
        categories=['Low', 'Medium', 'High'],
        ordered=True)


# In[ ]:


df = pd.read_csv('census.csv')
df = df[df['SUMLEV']==50]
df = df.set_index('STNAME').groupby(level=0)['CENSUS2010POP'].agg({'avg': np.average})
pd.cut(df['avg'],10)


# # Pivot Tables

# In[8]:


#http://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64
df = pd.read_csv('cars.csv')


# In[9]:


df.head()


# In[10]:


df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=np.mean)


# In[15]:


df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=[np.mean,np.min], margins=True)


# # Date Functionality in Pandas

# In[ ]:


import pandas as pd
import numpy as np


# ### Timestamp

# In[ ]:


pd.Timestamp('9/1/2016 10:05AM')


# ### Period

# In[ ]:


pd.Period('1/2016')


# In[ ]:


pd.Period('3/5/2016')


# ### DatetimeIndex

# In[ ]:


t1 = pd.Series(list('abc'), [pd.Timestamp('2016-09-01'), pd.Timestamp('2016-09-02'), pd.Timestamp('2016-09-03')])
t1


# In[ ]:


type(t1.index)
print(t1.index)
t1.index.to_period()


# ### PeriodIndex

# In[ ]:


t2 = pd.Series(list('def'), [pd.Period('2016-09'), pd.Period('2016-10'), pd.Period('2016-11')])
t2


# In[ ]:


type(t2.index)


# ### Converting to Datetime

# In[17]:


d1 = ['2 June 2013', 'Aug 29, 2014', '2015-06-26', '7/12/16']
ts3 = pd.DataFrame(np.random.randint(10, 100, (4,2)), index=d1, columns=list('ab'))
ts3


# In[18]:


ts3.index = pd.to_datetime(ts3.index)
ts3


# In[24]:


pd.to_datetime('4.7.12', dayfirst=True)
#pd.to_datetime(pd.Series(d1))
pd.to_datetime('4.7.12', dayfirst=False)


# ### Timedeltas

# In[ ]:


pd.Timestamp('9/3/2016')-pd.Timestamp('9/1/2016')


# In[ ]:


pd.Timestamp('9/2/2016 8:10AM') + pd.Timedelta('12D 3H')


# ### Working with Dates in a Dataframe

# In[25]:


dates = pd.date_range('10-01-2016', periods=9, freq='2W-SUN')
dates


# In[ ]:


df = pd.DataFrame({'Count 1': 100 + np.random.randint(-5, 10, 9).cumsum(),
                  'Count 2': 120 + np.random.randint(-5, 10, 9)}, index=dates)
df


# In[ ]:


df.index.weekday_name


# In[ ]:


df.diff()


# In[ ]:


df.resample('M').mean()


# In[ ]:


df['2017']


# In[ ]:


df['2016-12']


# In[ ]:


df['2016-12':]


# In[ ]:


df.asfreq('W', method='ffill')


# In[26]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df.plot()


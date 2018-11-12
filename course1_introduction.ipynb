
# coding: utf-8

# # Summary -- Introduction to Data Analysis with Python  2018-09
# ---
# 1. Sources
# 2. Read data
# 2. Data Info
# 2. Functions: map/lambda
# * Numpy
# 2. Merge Data
# 2. Datetime Resampling
# 4. Hyposysis

# ## Sources

# In[29]:


import pandas as pd
import numpy as np
import datetime as dt
import time as tm

get_ipython().run_line_magic('precision', '2')


# ## Read Data

# In[ ]:


df = pd.read_csv('cars.csv')


# In[ ]:


df = pd.read_excel('gdplev.xls')


# In[ ]:


df = pd.read_table('university_towns.txt')


# ## Data Info

# In[ ]:


# 普通变量
x = 2
type(x)


# In[ ]:


# data frame
df = pd.read_csv('cars.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.columns


# In[ ]:


df.index


# ## Datetime Resampling

# In[ ]:


# timestamp
tm.time()


# In[ ]:


dtnow = dt.datetime.fromtimestamp(tm.time())


# In[ ]:


str(dtnow.year) + "-" + str(dtnow.month)


# In[ ]:


dtnow + dt.timedelta(days = 10)


# In[38]:


df = pd.read_csv('City_Zhvi_AllHomes.csv')
df.set_index(['State', 'RegionName'], inplace=True)
df = df[[col for col in df.columns if '20' in col]]
df.columns = pd.to_datetime(df.columns)
df = df.resample('Q',axis=1).mean()
df.rename(columns=lambda x: str(x.to_period('Q')).lower(), inplace=True)
df


# ## Functions: map, lambda 

# In[ ]:


# map
store1 = [10.00, 11.00, 12.34, 2.34]
store2 = [9.00, 11.10, 12.34, 2.01]
cheapest = map(min, store1, store2)
for item in cheapest:
    print(item)


# In[ ]:


# lambda 
# todo


# ## Numpy

# In[ ]:


# 一维数组
np.array([2,3,5])


# In[ ]:


# 二维数组
arr = np.array([[2,3,4],[3,4,5]])


# In[ ]:


arr.shape


# In[ ]:


arr.dtype


# In[ ]:


arr.astype(float)


# create `special` arrays

# In[ ]:


np.ones((3,2))


# In[ ]:


np.zeros((3,2))


# In[ ]:


np.eye(3)


# In[ ]:


np.diag(arr)


# In[ ]:


np.arange(0, 30, 2) # start at 0 count up by 2, stop before 30


# In[ ]:


np.linspace(0, 4, 9) # return 9 evenly spaced values from 0 to 4


# In[ ]:


np.random.randint(0, 10, (4,3))


# In[31]:


np.random.binomial(20, .5, 10)


# In[34]:


np.random.uniform(0,1,10)


# `arange` returns evenly spaced values within a given interval.

# `reshape` reshapes non inplace, `resize` reshapes inplace

# In[ ]:


arr.reshape(3,2)
arr


# In[ ]:


arr.resize(3,2)
arr


# In[ ]:


arr.T


# In[ ]:


np.repeat([1, 2, 3], 3)


# In[ ]:


np.array([1, 2, 3] * 3)


# Use `vstack` to stack arrays in sequence vertically (row wise).

# In[ ]:


p = np.ones([2, 3], int)
p


# In[ ]:


np.vstack([p, 2*p])


# Use `hstack` to stack arrays in sequence horizontally (column wise).

# In[ ]:


np.hstack([p, 2*p])


# Iterate by row and index:

# In[ ]:


test = np.random.randint(0, 10, (2,3))
for i, row in enumerate(test):
    print('row', i, 'is', row)


# In[4]:


for i, j in zip(test, test**2):
    print(i,'+',j,'=',i+j)


# 判断`null`/`None`

# In[7]:


np.isnan(np.NaN)


# In[10]:


np.NaN == None


# In[11]:


np.NaN == np.NaN


# # pandas

# In[2]:


import pandas as pd


# `Series`

# In[3]:


animals = ['Tiger', 'Bear', None]
pd.Series(animals)


# In[15]:


sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s


# In[14]:


s.index


# In[17]:


s[1]


# In[18]:


s.iloc[1]


# In[19]:


s.loc['Golf']


# In[20]:


s['Golf']


# ## `group by` and `apply`

# In[21]:


census_df = pd.read_csv('census.csv')


# In[22]:


census_df.groupby(['STNAME'])['CTYNAME'].count().idxmax()


# ## `多行`语句

# In[26]:


(census_df[census_df['SUMLEV']==50]
 .groupby('STNAME')
 .apply(lambda x:x.nlargest(3, 'CENSUS2010POP'))
 .sum(level='STNAME')
 .nlargest(3, 'CENSUS2010POP'))


# ## 表关联 merge

# In[28]:


pd.merge(staff_df, student_df, how='outer', left_index=True, right_index=True)


# `Hyposysis` 假设检验

# In[43]:


from scipy import stats
stats.ttest_ind


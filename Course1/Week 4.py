
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# # Distributions in Pandas

# In[38]:


import pandas as pd
import numpy as np
x = np.random.binomial(20, .5, 1000)
(x>=11).mean()


# In[41]:


np.random.binomial(1, 0.5)


# In[40]:


np.random.binomial(1000, 0.5)/1000


# In[15]:


chance_of_tornado = 0.01/100
np.random.binomial(100000, chance_of_tornado,10)


# In[16]:


chance_of_tornado = 0.01

tornado_events = np.random.binomial(1, chance_of_tornado, 1000000)
    
two_days_in_a_row = 0
for j in range(1,len(tornado_events)-1):
    if tornado_events[j]==1 and tornado_events[j-1]==1:
        two_days_in_a_row+=1

print('{} tornadoes back to back in {} years'.format(two_days_in_a_row, 1000000/365))


# In[45]:


np.random.uniform(0,1)


# In[53]:


np.random.normal(0.75)


# Formula for standard deviation
# $$ \sigma = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \overline{x})^2}$$

# ## normal distribution
# $$ p(x) = \frac{1}{\sqrt{ 2 \pi \sigma^2 }}
#                  e^{ - \frac{ (x - \mu)^2 } {2 \sigma^2} } $$
#             

# In[60]:


distribution = np.random.normal(0.75,size=1000)

np.sqrt(np.sum((np.mean(distribution)-distribution)**2)/len(distribution))


# In[61]:


np.std(distribution)


# In[3]:


import numpy as np
import pandas as pd
s = pd.DataFrame(np.random.normal(0, 0.1, 100000))
s.hist(bins=1000)


# In[45]:


mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 100000)

# Verify the mean and the variance:
abs(mu - np.mean(s)) < 0.01
abs(sigma - np.std(s, ddof=1)) < 0.01

# P-P图
import matplotlib.pyplot as plt
count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
         np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r',label='a')
plt.show()


# ## 判断是否正太分布--QQ图 

# In[13]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
s = np.random.chisquare(500, 1000000)
print(len(s))
df = pd.DataFrame(s)
df.hist(bins=100)
#sm.qqplot(s, line='s')
print(stats.kstest(s, 'norm'))
print(stats.kurtosistest(s))


# ## 判断特征是否符合指定分布 stats.kstest
# * pvalue越大，说明符合指定分布
# * pvalue越小，小于阈值（如0.05），说明不符合正太分布

# In[1]:


# 判断是否符合指定分布
import scipy.stats as stats
import numpy as np
stats.kstest(np.random.normal(0, 1, 1000), 'norm')


# In[35]:


import pandas as pd
import random

##f.hist()
stats.kstest(np.random.rand(1000000), 'norm')
#np.random.rand?


# * kurtosis: 峰度：正态分布=3， >3 更尖，极值多
# * skew: 偏度

# In[105]:


#kurtosis: 峰度
stats.kurtosis(distribution)

# kurtosistest: pvalue大时正态分布, 小于0.05 不是正太分布
print(stats.kurtosistest(np.random.normal(0, 1, 1000)))
print(stats.kurtosistest(list(range(1000))))
stats.kurtosistest(distribution)


# In[66]:


chi_squared_df2 = np.random.chisquare(2, size=10000)
stats.skew(chi_squared_df2)


# In[103]:


chi_squared_df5 = np.random.chisquare(5, size=10000)
stats.skew(chi_squared_df5)
get_ipython().run_line_magic('pinfo', 'stats.chi2')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
mu = 0
sigma = 1
mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
X = np.random.normal(mu, sigma, 100000)
Y = X*X
Z = X*X*X
label='$(\mu=0,\sigma=1)^3$'


output = plt.hist([Z], bins=100, color='g',
                  label=['中文'])
plt.legend(loc='upper right')
print(stats.kurtosistest(Y))
plt.show()
#sm.qqplot(Y,line='s')
#stats.kurtosistest(np.exp(s))


# In[68]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt

output = plt.hist([chi_squared_df2,chi_squared_df5], bins=50, histtype='step', 
                  label=['2 degrees of freedom','5 degrees of freedom'])
plt.legend(loc='upper right')


# # Hypothesis Testing

# In[109]:


df = pd.read_csv('grades.csv')


# In[110]:


df.head()


# In[111]:


len(df)


# In[112]:


early = df[df['assignment1_submission'] <= '2015-12-31']
late = df[df['assignment1_submission'] > '2015-12-31']


# In[113]:


early.mean()


# In[114]:


late.mean()


# In[116]:


from scipy import stats
#stats.ttest_ind?


# In[117]:


stats.ttest_ind(early['assignment1_grade'], late['assignment1_grade'])


# In[118]:


stats.ttest_ind(early['assignment2_grade'], late['assignment2_grade'])


# In[119]:


stats.ttest_ind(early['assignment3_grade'], late['assignment3_grade'])


# In[121]:


stats.ttest_ind(early['assignment4_grade'], late['assignment4_grade'])


# In[122]:


stats.ttest_ind(early['assignment5_grade'], late['assignment5_grade'])


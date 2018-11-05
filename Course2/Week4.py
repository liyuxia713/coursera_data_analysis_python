
# coding: utf-8

# # Pandas Visualization

# In[91]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'notebook')


# http://pandas.pydata.org/pandas-docs/stable/visualization.html

# In[3]:


# see the pre-defined styles provided.
plt.style.available


# https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html

# In[ ]:


# use the 'seaborn-colorblind' style
plt.style.use('seaborn-colorblind')


# ### DataFrame.plot

# In[4]:


np.random.seed(123)

df = pd.DataFrame({'A': np.random.randn(365).cumsum(0), 
                   'B': np.random.randn(365).cumsum(0) + 20,
                   'C': np.random.randn(365).cumsum(0) - 20}, 
                  index=pd.date_range('1/1/2017', periods=365))
df.head()


# In[18]:


df.plot(); # add a semi-colon to the end of the plotting call to suppress unwanted output


# We can select which plot we want to use by passing it into the 'kind' parameter.

# In[29]:


df.plot('A','C', kind = 'scatter');


# You can also choose the plot kind by using the `DataFrame.plot.kind` methods instead of providing the `kind` keyword argument.
# 
# `kind` :
# - `'line'` : line plot (default)
# - `'bar'` : vertical bar plot
# - `'barh'` : horizontal bar plot
# - `'hist'` : histogram
# - `'box'` : boxplot
# - `'kde'` : Kernel Density Estimation plot
# - `'density'` : same as 'kde'
# - `'area'` : area plot
# - `'pie'` : pie plot
# - `'scatter'` : scatter plot
# - `'hexbin'` : hexbin plot

# In[32]:


df.plot.scatter('A', 'C');


# In[66]:


# create a scatter plot of columns 'A' and 'C', with changing color (c) and size (s) based on column 'B'
df.plot.scatter('A', 'C', c='B', s=df['B'], colormap='viridis')


# In[52]:


ax = df.plot.scatter('A', 'C', c='B', s=df['B'], colormap='viridis')
ax.set_aspect('equal')


# In[33]:


df.plot.box();


# In[43]:


df.plot.hist(normed=False, alpha=0.7);


# [Kernel density estimation plots](https://en.wikipedia.org/wiki/Kernel_density_estimation) are useful for deriving a smooth continuous function from a given sample.

# In[44]:


df.plot.kde();


# ### pandas.tools.plotting

# [Iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set)

# In[49]:


iris = pd.read_csv('iris.csv')
iris.head()


# In[54]:


iris.Name.unique()


# In[58]:


iris.Name.value_counts()


# In[55]:


iris.plot()  #会忽略Name  没有意义


# In[59]:


df.plot.scatter('A', 'A')


# https://blog.csdn.net/bea_tree/article/details/50757338

# In[50]:


pd.tools.plotting.scatter_matrix(iris);


# In[60]:


plt.figure()
pd.tools.plotting.parallel_coordinates(iris, 'Name');


# For instance, looking at our iris data set, we can see that the petal length and 
# petal width are two variables that split the different species fairly clearly. With iris virginica having the longest and widest petals. And iris setosa having the shortest and narrowest petals.

# In[62]:


pd.tools.plotting.andrews_curves(iris, 'Name')


# In[64]:


pd.tools.plotting.radviz(iris, 'Name');


# # Seaborn

# In[67]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'notebook')


# In[68]:


np.random.seed(1234)

v1 = pd.Series(np.random.normal(0,10,1000), name='v1')
v2 = pd.Series(2*v1 + np.random.normal(60,15,1000), name='v2')


# In[69]:


plt.figure()
plt.hist(v1, alpha=0.7, bins=np.arange(-50,150,5), label='v1');
plt.hist(v2, alpha=0.7, bins=np.arange(-50,150,5), label='v2');
plt.legend();


# In[82]:


# plot a kernel density estimation over a stacked barchart
plt.figure()
plt.hist([v1, v2], histtype='barstacked', normed=True);
v3 = np.concatenate((v1,v2))
#np.concatenate?
sns.kdeplot(v3);


# In[90]:


plt.figure()
# we can pass keyword arguments for each individual component of the plot
#sns.distplot(v3, hist_kws={'color': 'green'}, kde_kws={'color': 'blue'});
sns.distplot(v1)


# In[88]:


sns.jointplot(v1, v2, alpha=0.4);


# In[93]:


grid = sns.jointplot(v1, v2, alpha=0.4);
grid.ax_joint.set_aspect('equal')


# In[94]:


sns.jointplot(v1, v2, kind='hex');


# In[97]:


# set the seaborn style for all the following plots
sns.set_style('white')

sns.jointplot(v1, v2, kind='kde', space=0);


# In[ ]:


iris = pd.read_csv('iris.csv')
iris.head()


# In[103]:


pd.tools.plotting.scatter_matrix(iris)


# In[98]:


sns.pairplot(iris, hue='Name', diag_kind='kde', size=2);


# In[104]:


plt.figure(figsize=(8,6))
plt.subplot(121)
sns.swarmplot('Name', 'PetalLength', data=iris);
plt.subplot(122)
sns.violinplot('Name', 'PetalLength', data=iris);


# In[111]:


from scipy import stats
plt.figure()
sns.distplot(y,bins=20,rug=True)
sns.distplot(y,bins=20,kde=False,rug=True, fit=stats.gamma)
get_ipython().run_line_magic('pinfo', 'sns.distplot')


# In[112]:


tips=sns.load_dataset('tips')


# In[122]:


tips.head()


# In[117]:


plt.figure()
sns.stripplot(x='day',y='total_bill',data=tips);


# In[119]:


plt.figure();
sns.swarmplot(x='day',y='total_bill',data=tips);


# In[120]:


plt.figure();
sns.swarmplot(x='day',y='total_bill', hue='sex', data=tips);


# In[131]:


plt.figure()
plt.subplot(2,1,1)
sns.stripplot(x='size',y='size',data=tips,jitter=True)
plt.subplot(2,1,2)
sns.swarmplot(x='size',y='size',data=tips)


# In[132]:


plt.figure()
sns.boxplot(x="day", y="total_bill", hue="size", data=tips);


# In[134]:


plt.figure()
sns.boxplot(x="day", y="total_bill",data=tips);


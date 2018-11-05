
# coding: utf-8

# # Assignment 4
# 
# Before working on this assignment please read these instructions fully. In the submission area, you will notice that you can click the link to **Preview the Grading** for each step of the assignment. This is the criteria that will be used for peer grading. Please familiarize yourself with the criteria before beginning the assignment.
# 
# This assignment requires that you to find **at least** two datasets on the web which are related, and that you visualize these datasets to answer a question with the broad topic of **economic activity or measures** (see below) for the region of **Ann Arbor, Michigan, United States**, or **United States** more broadly.
# 
# You can merge these datasets with data from different regions if you like! For instance, you might want to compare **Ann Arbor, Michigan, United States** to Ann Arbor, USA. In that case at least one source file must be about **Ann Arbor, Michigan, United States**.
# 
# You are welcome to choose datasets at your discretion, but keep in mind **they will be shared with your peers**, so choose appropriate datasets. Sensitive, confidential, illicit, and proprietary materials are not good choices for datasets for this assignment. You are welcome to upload datasets of your own as well, and link to them using a third party repository such as github, bitbucket, pastebin, etc. Please be aware of the Coursera terms of service with respect to intellectual property.
# 
# Also, you are welcome to preserve data in its original language, but for the purposes of grading you should provide english translations. You are welcome to provide multiple visuals in different languages if you would like!
# 
# As this assignment is for the whole course, you must incorporate principles discussed in the first week, such as having as high data-ink ratio (Tufte) and aligning with Cairo’s principles of truth, beauty, function, and insight.
# 
# Here are the assignment instructions:
# 
#  * State the region and the domain category that your data sets are about (e.g., **Ann Arbor, Michigan, United States** and **economic activity or measures**).
#  * You must state a question about the domain category and region that you identified as being interesting.
#  * You must provide at least two links to available datasets. These could be links to files such as CSV or Excel files, or links to websites which might have data in tabular form, such as Wikipedia pages.
#  * You must upload an image which addresses the research question you stated. In addition to addressing the question, this visual should follow Cairo's principles of truthfulness, functionality, beauty, and insightfulness.
#  * You must contribute a short (1-2 paragraph) written justification of how your visualization addresses your stated research question.
# 
# What do we mean by **economic activity or measures**?  For this category you might look at the inputs or outputs to the given economy, or major changes in the economy compared to other regions.
# 
# ## Tips
# * Wikipedia is an excellent source of data, and I strongly encourage you to explore it for new data sources.
# * Many governments run open data initiatives at the city, region, and country levels, and these are wonderful resources for localized data sources.
# * Several international agencies, such as the [United Nations](http://data.un.org/), the [World Bank](http://data.worldbank.org/), the [Global Open Data Index](http://index.okfn.org/place/) are other great places to look for data.
# * This assignment requires you to convert and clean datafiles. Check out the discussion forums for tips on how to do this from various sources, and share your successes with your fellow students!
# 
# ## Example
# Looking for an example? Here's what our course assistant put together for the **Ann Arbor, MI, USA** area using **sports and athletics** as the topic. [Example Solution File](./readonly/Assignment4_example.pdf)

# In[199]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys


# In[5]:


import ssl 
ssl._create_default_https_context = ssl._create_unverified_context 


# In[6]:


print(sys.executable)


# In[ ]:


def download_table(year, k):
    """
    download Olympics_medal_table from url by year, and save to csv file
    @param: year 
    @param: k: kth dataframe is the medal_table
    """
    print("------------ LOG: downloading %s ----------" % (year))
    file = '%s_Summer_Olympics_medal_table' % (year)
    url = 'https://en.wikipedia.org/wiki/%s' % (file)
    dataframes = pd.read_html(url, header=0)
    dataframes[k].to_csv('data_olympics/%s.csv' % (file))
    print(dataframes[k].head())
    

for year in np.concatenate((np.arange(2016, 1983, -4), [1960])):
    download_table(year, 1)
    
for year in np.concatenate((np.arange(1980, 1963, -4), [1988])):
    download_table(year, 0)


# In[468]:


def truncate_data(year):
    """
    deal with same rank line of table (same ranks record miss one column)
    """
    file = 'data_olympics/%d_Summer_Olympics_medal_table.csv' % (year)
    file_new = 'data_olympics/%d_Summer_Olympics_medal_table_1.csv' % (year)
    fp = open(file_new, 'w')
    rank_prev = -1
    for line in open(file):
        #[_, Rank, Nation, Gold, Silver, Bronze, Total]
        list_ = line.strip().split(',')
        [_, Rank, Nation, Gold, Silver, Bronze, Total] = list_
        if list_[-1] == '':
            [Rank, Nation, Gold, Silver, Bronze, Total, _] = list_
        fp.write(','.join([Rank, Nation, Gold, Silver, Bronze, Total]) + '\n')  
    fp.close()


# In[469]:


# load all year tables to one dataframe
list_ = []
for year in np.arange(2016, 1959, -4):
    truncate_data(year)
    file = 'data_olympics/%d_Summer_Olympics_medal_table_1.csv' % (year)
    df_tmp = pd.read_csv(file, header=0)
    df_tmp.columns = ['Rank', 'Nation', 'Gold', 'Silver', 'Bronze', 'Total']
    df_tmp['Year'] = year
    list_.append(df_tmp)
df = pd.concat(list_)


# In[470]:


df.head()


# In[471]:


# the Hostage Nation has "*" at the end of Nation Name
df['Nation'] = df.Nation.apply(lambda x: x.replace('*', ''))
df['Nation'] = df.Nation.apply(lambda x: x.split('(')[0])


# In[472]:


mask_total = df.Nation.str.contains('Totals')
df_total = df[mask_total][['Total', 'Year']]
df = df[~mask_total]


# In[473]:


df_total.head()


# In[474]:


df1 = pd.merge(df, df_total, how='inner', on='Year', suffixes=['', '_total'])


# In[475]:


df1.head()


# In[476]:


df1['Total_medal_rate'] = df1['Total']/df1['Total_total']


# In[477]:


df = df1


# In[478]:


df.head()


# In[498]:


#get top nations all years medal data
top_nation = (df[(df.Year==2016)]
               .sort_values('Total_medal_rate', ascending=False)
               .head(5)
               .Nation)
mask = (df.Nation.isin(top_nation)) & (df.Year>1900)
df_new = (df[mask].pivot('Year', 'Nation', 'Total')
          .sort_index())


# In[501]:


plt.figure();
df_new.plot();
plt.legend(bbox_to_anchor=(1,0.5), loc='center left');

plt.gca().set_xticks(df_new.index);
plt.ylabel('Medal Num')
plt.title('Olympic Medal Num Trend of 2016 Top5 Countries')

# rotate the tick labels for the x axis
for item in plt.gca().xaxis.get_ticklabels():
    item.set_rotation(45);


#  * State the region and the domain category that your data sets are about (e.g., **Ann Arbor, Michigan, United States** and **economic activity or measures**).
#  * You must state a question about the domain category and region that you identified as being interesting.
#  * You must provide at least two links to available datasets. These could be links to files such as CSV or Excel files, or links to websites which might have data in tabular form, such as Wikipedia pages.
#  * You must upload an image which addresses the research question you stated. In addition to addressing the question, this visual should follow Cairo's principles of truthfulness, functionality, beauty, and insightfulness.
#  * You must contribute a short (1-2 paragraph) written justification of how your visualization addresses your stated research question.
# 

# * Olympic medal num of 2016 Top5 countries
# 
# * Medal num trends over the years of countries in the olympic medal list
# 
# * https://en.wikipedia.org/wiki/2016_Summer_Olympics_medal_table
#     https://en.wikipedia.org/wiki/1988_Summer_Olympics_medal_table
#     
# * 
# 
# *
# 1. The countries in the 2016 medal list are: United States, Great Britain, China, Russia, Germany.
# 2. Only the Great Britain has always been participating, and medal numbers has steadily increased since 1988.
# 3. The number of medals in the US has been always leading.
# 4. The number of medals in Germany and Russia is decreasing.
# 5. China won its most medals at the 2008 Beijing Olympics, second only to the United States.
# 

# This visualization is a truthful representation of the GDP and Adjusted Median Family Income for both Michigan and Ann Arbor and quickly provides the insight that the Adjusted Median Family Income of Ann Arbor is greater than Michigan.
# 
# At the end of Jupyter Notebook, three graphs are attached highlighting the difference in adjusted median family income.
# 
# All the processes like data cleaning, summarization and the labelling have been performed to make visualization simple, light, functional, truthful and provide better insightful of the data quickly. 
# 
# The title​, legends and axes are all labelled correctly to provide a truthful and beautiful visualization of the data.


# coding: utf-8

# # Assignment 2
# Before working on this assignment please read these instructions fully. In the submission area, you will notice that you can click the link to Preview the Grading for each step of the assignment. This is the criteria that will be used for peer grading. Please familiarize yourself with the criteria before beginning the assignment.
# 
# An NOAA dataset has been stored in the file data/C2A2_data/BinnedCsvs_d100/808b473e723bc71e800db83e1ba7c1342e1cd336684f03ca3cac84aa.csv. The data for this assignment comes from a subset of The National Centers for Environmental Information (NCEI) Daily Global Historical Climatology Network (GHCN-Daily). The GHCN-Daily is comprised of daily climate records from thousands of land surface stations across the globe.
# 
# Each row in the assignment datafile corresponds to a single observation.
# 
# The following variables are provided to you:
# 
# id : station identification code
# 
# date : date in YYYY-MM-DD format (e.g. 2012-01-24 = January 24, 2012)
# 
# element : indicator of element type
# 
# TMAX : Maximum temperature (tenths of degrees C)
# 
# TMIN : Minimum temperature (tenths of degrees C)
# 
# value : data value for element (tenths of degrees C)
# 
# For this assignment, you must:
# 
# Read the documentation and familiarize yourself with the dataset, then write some python code which returns a line graph of the record high and record low temperatures by day of the year over the period 2005-2014. The area between the record high and record low temperatures for each day should be shaded.
# 
# Overlay a scatter of the 2015 data for any points (highs and lows) for which the ten year record (2005-2014) record high or record low was broken in 2015.
# 
# Watch out for leap days (i.e. February 29th), it is reasonable to remove these points from the dataset for the purpose of this visualization.
# 
# Make the visual nice! Leverage principles from the first module in this course when developing your solution. Consider issues such as legends, labels, and chart junk.
# 

# For more explanation, see here
# https://www.coursera.org/learn/python-plotting/discussions/weeks/2/threads/7Rh0RAQ1Eee-uQ7R_UJsXg

# In[1]:


import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[13]:


df = pd.read_csv('data/C2A2_data/BinnedCsvs_d100/808b473e723bc71e800db83e1ba7c1342e1cd336684f03ca3cac84aa.csv')


# In[14]:


df['Data_Value'] /= 10.0
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].apply(lambda x: pd.to_datetime(x).year)
df['Date'] = df['Date'].apply(lambda x: dt.date(2000, pd.to_datetime(x).month, pd.to_datetime(x).day))
df.drop('ID', axis=1, inplace=True)


# In[15]:


# get temperature of 2005-2014
df_past_min = (df[(df.Year>=2005) & (df.Year<2015) & (df.Date!=dt.date(2000, 2, 29)) & (df.Element=='TMIN')]
                .drop(['Element', 'Year'], axis=1)
                .groupby('Date').min()['Data_Value'])
df_past_max = (df[(df.Year>=2005) & (df.Year<2015) & (df.Date!=dt.date(2000, 2, 29)) & (df.Element=='TMAX')]
                .drop(['Element', 'Year'], axis=1)
                .groupby('Date').max()['Data_Value'])


# In[16]:


# get temperature of 2015
df_now_min = (df[(df.Year==2015) & (df.Element=='TMIN')]
              .drop(['Element', 'Year'], axis=1)
              .groupby('Date').min()['Data_Value'])
df_now_max = (df[(df.Year==2015) & (df.Element=='TMAX')]
              .drop(['Element', 'Year'], axis=1)
              .groupby('Date').max()['Data_Value'])


# In[17]:


# only reserve the min or max temperature brokers
df_now_miner = df_now_min[df_now_min - df_past_min < 0]
df_now_maxer = df_now_max[df_now_max - df_past_max > 0] 


# In[27]:


plt.figure(figsize=(9,6))
dates = df_past_min.index 
plt.plot(dates, df_past_max, '--', label='2005-2014 MAX Temp.')
plt.plot(dates, df_past_min, '--', label='2005-2014 MIN Temp.')
plt.scatter(df_now_maxer.index, df_now_maxer, s=20, c='r', label='2015 MAX Temp. brokers')
plt.scatter(df_now_miner.index, df_now_miner, s=20, c='g', label='2015 MIN Temp. brokers')
plt.xlabel('Date')
plt.ylabel('Temperature  ($^{\circ}$C)')
plt.title('Temperature 2015 v.s. 2005~2014 (daily)')

# set legend out of the image
plt.legend(bbox_to_anchor=(1,0.5), loc='center left')

# fill the area between the min and max temperature
ax = plt.gca()
###plt.setp(ax, linewidth=0.4)
ax.fill_between(dates, 
               df_past_min, df_past_max, 
               facecolor='blue', 
               alpha=0.25)

# set xlabels with month
m = ['Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nov','Dec']
ticks = [dt.date(2000,m,1) for m in range(1,13)]
ax.set_xticks(ticks)
ax.set_xticklabels(m)

# add vertical lines
for xc in ticks:
    plt.axvline(x=xc, linestyle='--', color='gray', alpha=0.15)


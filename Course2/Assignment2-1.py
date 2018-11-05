
# coding: utf-8

# # Assignment 2
# 
# Before working on this assignment please read these instructions fully. In the submission area, you will notice that you can click the link to **Preview the Grading** for each step of the assignment. This is the criteria that will be used for peer grading. Please familiarize yourself with the criteria before beginning the assignment.
# 
# An NOAA dataset has been stored in the file `data/C2A2_data/BinnedCsvs_d400/93224eb643a4e7f4d2da8084ef34d9244c6b39dfd5cf24002d793096.csv`. The data for this assignment comes from a subset of The National Centers for Environmental Information (NCEI) [Daily Global Historical Climatology Network](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt) (GHCN-Daily). The GHCN-Daily is comprised of daily climate records from thousands of land surface stations across the globe.
# 
# Each row in the assignment datafile corresponds to a single observation.
# 
# The following variables are provided to you:
# 
# * **id** : station identification code
# * **date** : date in YYYY-MM-DD format (e.g. 2012-01-24 = January 24, 2012)
# * **element** : indicator of element type
#     * TMAX : Maximum temperature (tenths of degrees C)
#     * TMIN : Minimum temperature (tenths of degrees C)
# * **value** : data value for element (tenths of degrees C)
# 
# For this assignment, you must:
# 
# 1. Read the documentation and familiarize yourself with the dataset, then write some python code which returns a line graph of the record high and record low temperatures by day of the year over the period 2005-2014. The area between the record high and record low temperatures for each day should be shaded.
# 2. Overlay a scatter of the 2015 data for any points (highs and lows) for which the ten year record (2005-2014) record high or record low was broken in 2015.
# 3. Watch out for leap days (i.e. February 29th), it is reasonable to remove these points from the dataset for the purpose of this visualization.
# 4. Make the visual nice! Leverage principles from the first module in this course when developing your solution. Consider issues such as legends, labels, and chart junk.
# 
# The data you have been given is near **Saint Paul, Minnesota, United States**, and the stations the data comes from are shown on the map below.

# In[1]:


import matplotlib.pyplot as plt
import mplleaflet
import pandas as pd

def leaflet_plot_stations(binsize, hashid):

    df = pd.read_csv('data/C2A2_data/BinSize_d{}.csv'.format(binsize))

    station_locations_by_hash = df[df['hash'] == hashid]

    lons = station_locations_by_hash['LONGITUDE'].tolist()
    lats = station_locations_by_hash['LATITUDE'].tolist()

    plt.figure(figsize=(8,8))

    plt.scatter(lons, lats, c='r', alpha=0.7, s=200)

    return mplleaflet.display()

leaflet_plot_stations(400,'93224eb643a4e7f4d2da8084ef34d9244c6b39dfd5cf24002d793096')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[ ]:


import numpy as np


# In[ ]:


import matplotlib as mpl
mpl.get_backend()


# ```python
# hashid = "93224eb643a4e7f4d2da8084ef34d9244c6b39dfd5cf24002d793096"
# msp_file = "data/C2A2_data/BinnedCsvs_d400/" + hashid + ".csv"
# # df = pd.read_csv(msp_file)
# df = pd.read_csv(msp_file, 
#                  parse_dates=True)
# ```

# In[ ]:


import matplotlib.dates as mdates
from matplotlib.ticker import EngFormatter, StrMethodFormatter
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as ticker


# In[ ]:


msp_file = "data/C2A2_data/BinnedCsvs_d400/93224eb643a4e7f4d2da8084ef34d9244c6b39dfd5cf24002d793096.csv"
df = pd.read_csv(msp_file, 
                 parse_dates=True)


# In[ ]:


df['DateTime'] = pd.to_datetime(df['Date'], 
                                format="%Y-%m-%d")


# In[ ]:


# df = df.set_index('DateTime')
df.index = df['DateTime']


# In[ ]:


df.columns


# Filtering historical data (2005:2014)

# In[ ]:


df_hist = df['2005':'2014']


# Getting new data 2015

# In[ ]:


df_new = df['2015']


# Working with `TMAX Element`

# In[ ]:


tmax_df = df_hist[df_hist['Element'] == 'TMAX']


# In[ ]:


tmax_df = (tmax_df.groupby(by=tmax_df.index.dayofyear)
                  .max())


# In[ ]:


# tmax_df.index = tmax_df['DateTime']
tmax_df = tmax_df.set_index('DateTime')


# In[ ]:


tmax_plot = tmax_df['2014']
tmax_plot = tmax_plot.reset_index()


# In[ ]:


# tmax_plot['MonthDay'] = pd.DatetimeIndex(tmax_plot['Date'])
tmax_plot['MonthDay'] = pd.to_datetime(tmax_plot['Date'])

# tmax_plot['MonthDay'] = list(map(lambda x: x.strftime("%b-%d"), tmax_plot['MonthDay']))
# tmax_plot['MonthDay'] = list(map(lambda x: x.strftime("%b"), tmax_plot['MonthDay']))
# tmax_plot['MonthDay'] = (pd.DatetimeIndex(tmax_plot['Date'])
#                            .map(lambda x: x.strftime("%b-%d")))

tmax_plot['MonthDay'] = tmax_plot['MonthDay'].map(lambda x: x.replace(year=2015))

# tmax_plot['MonthDay2'] = np.arange('2014-01-01', '2014-12-31', dtype='datetime64[D]')
# tmax_plot['MonthDay2'] = list(pd.to_datetime, tmax_plot['MonthDay2'])


# In[ ]:


# tmax_plot['Data_Value'] = tmax_plot['Data_Value'] / 10
tmax_plot['Data_Value'] = tmax_plot['Data_Value'] * 0.10


# Working with `TMIN Element`

# In[ ]:


tmin_df = df_hist[df_hist['Element'] == 'TMIN']


# In[ ]:


tmin_df = (tmin_df.groupby(by=tmin_df.index.dayofyear)
                  .min())


# In[ ]:


# tmax_df.index = tmax_df['DateTime']
tmin_df = tmin_df.set_index('DateTime')


# In[ ]:


tmin_plot = tmin_df['2005']
tmin_plot = tmin_plot.reset_index()


# In[ ]:


# tmin_plot['MonthDay'] = pd.DatetimeIndex(tmin_plot['Date'])
tmin_plot['MonthDay'] = pd.to_datetime(tmin_plot['Date'])

# tmin_plot['MonthDay'] = list(map(lambda x: x.strftime("%b-%d"), tmin_plot['MonthDay']))
# tmin_plot['MonthDay'] = list(map(lambda x: x.strftime("%b"), tmin_plot['MonthDay']))

tmin_plot['MonthDay'] = tmin_plot['MonthDay'].map(lambda x: x.replace(year=2015))
# tmin_plot['MonthDay'] = list(map(lambda x: x.strftime("%b"), tmin_plot['MonthDay']))

# tmax_plot['MonthDay'] = (pd.DatetimeIndex(tmax_plot['Date'])
#                            .map(lambda x: x.strftime("%b-%d")))


# In[ ]:


# tmin_plot['Data_Value'] = tmin_plot['Data_Value'] / 10
tmin_plot['Data_Value'] = tmin_plot['Data_Value'] * 0.10


# In[ ]:


# observation_dates = np.arange('2014-01-01', '2014-12-31', dtype='datetime64[D]')
# observation_dates = list(map(pd.to_datetime, observation_dates))
observation_dates = list(map(pd.to_datetime, tmax_plot['MonthDay']))


# Working with `TMAX Element` in new data 2015

# In[ ]:


newtmax_df = df_new[df_new['Element'] == "TMAX"]


# In[ ]:


newtmax_df = (newtmax_df.groupby(newtmax_df.index.dayofyear)
                        .max())


# In[ ]:


newtmax_df = newtmax_df.reset_index()


# In[ ]:


newtmax_df["MonthDay"] = pd.to_datetime(newtmax_df['Date'])


# In[ ]:


newtmax_plot = newtmax_df.copy()


# In[ ]:


# newtmax_plot['Data_Value'] = newtmax_plot['Data_Value'] / 10
newtmax_plot['Data_Value'] = newtmax_plot['Data_Value'] * 0.10


# In[ ]:


newtmax_plot = newtmax_plot[newtmax_plot['Data_Value'] >= tmax_plot['Data_Value']]


# Working with `TMIN Element` in new data 2015

# In[ ]:


newtmin_df = df_new[df_new['Element'] == 'TMIN']


# In[ ]:


newtmin_df = (newtmin_df.groupby(newtmin_df.index.dayofyear)
                        .min())


# In[ ]:


newtmin_df = newtmin_df.reset_index()


# In[ ]:


newtmin_df["MonthDay"] = pd.to_datetime(newtmin_df['Date'])


# In[ ]:


newtmin_plot = newtmin_df.copy()


# In[ ]:


# newtmin_plot['Data_Value'] = newtmin_plot['Data_Value'] / 10
newtmin_plot['Data_Value'] = newtmin_plot['Data_Value'] * 0.1


# In[ ]:


newtmin_plot = newtmin_plot[newtmin_plot['Data_Value'] <= tmin_plot['Data_Value']]


# In[ ]:


# observation_dates = np.arange('2014-01-01', '2014-12-31', dtype='datetime64[D]')
# observation_dates = list(map(pd.to_datetime, observation_dates))
observation_dates = list(map(pd.to_datetime, tmax_plot['MonthDay']))
point_observation_dates = list(map(pd.to_datetime, newtmax_plot['MonthDay']))
point_observation_dates_min = list(map(pd.to_datetime, newtmin_plot['MonthDay']))


# Main Plot for the Assignment

# In[ ]:


plt.figure(figsize=(9,5))

# plt.plot(tmax_plot['MonthDay'], tmax_plot['Data_Value'], '-k', lw=1)
# plt.plot(tmin_plot['MonthDay'], tmin_plot['Data_Value'], '-k', lw=1)
# plt.plot(tmax_plot['Data_Value'], '-k', lw=1)
# plt.plot(tmin_plot['Data_Value'], '-k', lw=1)
plt.plot(observation_dates, tmax_plot['Data_Value'], 
         '-k', lw=1, alpha=0.3,
         # label='Historical Max',
         label='_nolegend_')
plt.plot(observation_dates, tmin_plot['Data_Value'], 
         '-k', lw=1, alpha=0.3,
         # label='Historical Min',
         label='_nolegend_')
plt.scatter(point_observation_dates, newtmax_plot['Data_Value'],
            s=15, color='darkred', alpha=0.6,
            label='Broken Record High')
plt.scatter(point_observation_dates_min, newtmin_plot['Data_Value'],
            s=15, color='#1F77B4', alpha=0.8,
            label='Broken Record Low')
# plt.fill_between()
plt.show()

# title and labels
# plt.xlabel("Year")
# plt.ylabel("Temperature in Celsius")
plt.title("Record Temperatures in 2015 Relative to 2005-2014 Years: Saint Paul (Minnesota) Area")

# fill area between two lines
# fill the area between the linear data and exponential data
# plt.gca().fill_between(range(len(tmax_plot['Data_Value'])), 
#                        tmin_plot['Data_Value'], tmax_plot['Data_Value'],
#                        facecolor='grey',
#                        alpha=0.5)
plt.gca().fill_between(observation_dates, 
                       tmin_plot['Data_Value'], tmax_plot['Data_Value'],
                       facecolor='lightslategrey',
                       alpha=0.4,
                       # label='2005-2014 Temperature Ranges (Record High-Low)'
                       label='Daily Historical Temperature Ranges: Record High-Low')

# removing axis ticks
plt.tick_params(left="off", bottom="off")
# plt.tick_params(left="off")
# plt.tick_params(bottom="off")

# getting axis artists
ax = plt.gca()

# # removing frame
# for spine in plt.gca().spines.values():
#     spine.set_visible(False)
# removing all frame except Y axis
ax.spines["left"].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.spines["bottom"].set_visible(False)
    
# editing x-axis (date)
# import matplotlib.dates as mdates
# showing ONLY months with three letters
myFmt = mdates.DateFormatter('%b')
ax.xaxis.set_major_formatter(myFmt)

# adding Celsius to Y axis number
# from matplotlib.ticker import EngFormatter, StrMethodFormatter
ax.yaxis.set_major_formatter(EngFormatter(unit=u"°C"))
# ax.yaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f} °C"))

# adding verticle grid (minor)
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.ticker import AutoMinorLocator

# minor_locator = AutoMinorLocator(2)
# ax.xaxis.set_minor_locator(minor_locator)
# plt.grid(which='minor')
# # adjusting minor grid style
# ax.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
# ignoring minor grid
# minor_locator = AutoMinorLocator(2)
# ax.xaxis.set_minor_locator(minor_locator)
plt.grid(which='major')
# adjusting minor grid style
ax.grid(which='major', linestyle=':', linewidth='0.5', color='grey')
# displaying major grid only vertically
# ax.xaxis.grid(True)
ax.yaxis.grid(False)

# moving x label right
# plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left" )
plt.setp(ax.xaxis.get_majorticklabels(), ha="left" )

# editing X axis to include all months
# ax.xaxis.set_major_locator(plt.MultipleLocator(35))
ax.set_xlim(['2015-01-01', '2015-12-31'])

# # editing Y axis step size
# # import matplotlib.ticker as ticker
# ax = plt.axes()
ax.yaxis.set_major_locator(ticker.MultipleLocator(40))
# # ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

# adding appropriate legends
# plt.legend(['Baseline', 'Competition', 'Us'])
# plt.legend()
# plt.legend().set_linewidth(0.0)
plt.legend(frameon=False)

# plt.show()
# ax.grid(True)

plt.tight_layout()    

plt.show()


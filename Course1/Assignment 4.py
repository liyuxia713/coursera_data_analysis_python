
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


# # Assignment 4 - Hypothesis Testing
# This assignment requires more individual learning than previous assignments - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.
# 
# Definitions:
# * A _quarter_ is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
# * A _recession_ is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
# * A _recession bottom_ is the quarter within a recession which had the lowest GDP.
# * A _university town_ is a city which has a high percentage of university students compared to the total population of the city.
# 
# **Hypothesis**: University towns have their mean housing prices less effected by recessions. Run a t-test to compare the ratio of the mean price of houses in university towns the quarter before the recession starts compared to the recession bottom. (`price_ratio=quarter_before_recession/recession_bottom`)
# 
# The following data files are available for this assignment:
# * From the [Zillow research data site](http://www.zillow.com/research/data/) there is housing data for the United States. In particular the datafile for [all homes at a city level](http://files.zillowstatic.com/research/public/City/City_Zhvi_AllHomes.csv), ```City_Zhvi_AllHomes.csv```, has median home sale prices at a fine grained level.
# * From the Wikipedia page on college towns is a list of [university towns in the United States](https://en.wikipedia.org/wiki/List_of_college_towns#College_towns_in_the_United_States) which has been copy and pasted into the file ```university_towns.txt```.
# * From Bureau of Economic Analysis, US Department of Commerce, the [GDP over time](http://www.bea.gov/national/index.htm#gdp) of the United States in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in the file ```gdplev.xls```. For this assignment, only look at GDP data from the first quarter of 2000 onward.
# 
# Each function in this assignment below is worth 10%, with the exception of ```run_ttest()```, which is worth 50%.

# In[2]:


# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}


# In[3]:


def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the 
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
    columns=["State", "RegionName"]  )
    
    The following cleaning needs to be done:

    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character '\n'. '''
    df = pd.read_table('university_towns.txt', header=None, names=['State'])
    df['RegionName'] = df['State'].apply(lambda x: x.split('(')[0].strip() if x.count('[ed') <= 0 else '')
    df['State'] = df['State'].apply(lambda x: x.split('[ed')[0].strip() if x.count('[ed') > 0 else np.NaN).fillna(method="ffill")
    #df.replace({'State': dict(zip(states.values(), states.keys()))}, inplace=True)
    df = df[df.RegionName != '']
    return df.reset_index(drop=True)
#get_list_of_university_towns()


# In[4]:


def get_dataframe_gdp():
    df = pd.read_excel('gdplev.xls', usecols=[4,6], skiprows=4, names=['Quarter', 'GDP'])
    df = df[df['Quarter']>'2000']
    return df

def get_recession_start():
    '''Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3'''
    df = get_dataframe_gdp()
    df['GDP_prev'] = df['GDP'].shift(1)
    df['GDP_next'] = df['GDP'].shift(-1)
    df['recession_start'] = df.apply(lambda x: 1 if x.GDP < x.GDP_prev and x.GDP > x.GDP_next else 0, axis=1)
    return df[df['recession_start']==1]['Quarter'].iloc[0]
#get_recession_start()


# In[5]:


def get_recession_end():
    '''Returns the year and quarter of the recession end time as a 
    string value in a format such as 2005q3'''
    df = get_dataframe_gdp()
    recession_start = get_recession_start()
    df['GDP_prev'] = df['GDP'].shift(1)
    df['GDP_pprev'] = df['GDP'].shift(2)
    df['recession_end'] = df.apply(lambda x: 1 if x.Quarter > recession_start and x.GDP > x.GDP_prev and x.GDP_prev > x.GDP_pprev else 0, axis=1)
    return df[df['recession_end']==1]['Quarter'].iloc[0]


# In[6]:


def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a 
    string value in a format such as 2005q3'''
    recession_start = get_recession_start()
    recession_end = get_recession_end()
    df = get_dataframe_gdp()
    df = df[(df.Quarter>=recession_start) & (df.Quarter<=recession_end)]
    df.set_index('Quarter', inplace=True)
    return df['GDP'].astype(float).idxmin(axis=1)


# In[7]:


def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
    df = pd.read_csv('City_Zhvi_AllHomes.csv')
    df.replace({'State': states}, inplace=True)
    df.set_index(['State', 'RegionName'], inplace=True)
    df = df[[col for col in df.columns if '20' in col]]
    # 转为日期
    # print(df.columns): Index(['2000-01',...], dtype='object', length=200)
    # print(df.columns): DatetimeIndex(['2000-01-01',...], dtype='datetime64[ns]', length=200, freq=None)
    df.columns = pd.to_datetime(df.columns)

    df = df.resample('Q',axis=1).mean()
    df.rename(columns=lambda x: str(x.to_period('Q')).lower(), inplace=True)
    return df


# In[12]:


def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
    from datetime import timedelta
    #before recession start
    start = pd.to_datetime(get_recession_start()) - timedelta(1)
    start = str(start.to_period('Q')).lower()
    bottom = get_recession_bottom()

    housing = convert_housing_data_to_quarters()[[start, bottom]]
    housing['ratio'] = housing[start] / housing[bottom]

    university = get_list_of_university_towns()
    university.set_index(['State', 'RegionName'], inplace=True)
    f = housing.index.isin(university.index)
    uni = housing.loc[f==True]  
    non_uni = housing.loc[f==False]

    better = 'non university town' if uni['ratio'].mean() > non_uni['ratio'].mean() else 'university town'
    (statistics, pvalue) = ttest_ind(uni['ratio'], non_uni['ratio'], nan_policy='omit')
    return ((pvalue < 0.01), pvalue, better)


# In[ ]:


#get_dataframe_gdp().to_csv('my_gdp.csv')


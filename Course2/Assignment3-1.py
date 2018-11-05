
# coding: utf-8

# # Assignment 3 - Building a Custom Visualization
# 
# ---
# 
# In this assignment you must choose one of the options presented below and submit a visual as well as your source code for peer grading. The details of how you solve the assignment are up to you, although your assignment must use matplotlib so that your peers can evaluate your work. The options differ in challenge level, but there are no grades associated with the challenge level you chose. However, your peers will be asked to ensure you at least met a minimum quality for a given technique in order to pass. Implement the technique fully (or exceed it!) and you should be able to earn full grades for the assignment.
# 
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ferreira, N., Fisher, D., & Konig, A. C. (2014, April). [Sample-oriented task-driven visualizations: allowing users to make better, more confident decisions.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Ferreira_Fisher_Sample_Oriented_Tasks.pdf) 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (pp. 571-580). ACM. ([video](https://www.youtube.com/watch?v=BI7GAs-va-Q))
# 
# 
# In this [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Ferreira_Fisher_Sample_Oriented_Tasks.pdf) the authors describe the challenges users face when trying to make judgements about probabilistic data generated through samples. As an example, they look at a bar chart of four years of data (replicated below in Figure 1). Each year has a y-axis value, which is derived from a sample of a larger dataset. For instance, the first value might be the number votes in a given district or riding for 1992, with the average being around 33,000. On top of this is plotted the 95% confidence interval for the mean (see the boxplot lectures for more information, and the yerr parameter of barcharts).
# 
# <br>
# <img src="readonly/Assignment3Fig1.png" alt="Figure 1" style="width: 400px;"/>
# <h4 style="text-align: center;" markdown="1">  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Figure 1 from (Ferreira et al, 2014).</h4>
# 
# <br>
# 
# A challenge that users face is that, for a given y-axis value (e.g. 42,000), it is difficult to know which x-axis values are most likely to be representative, because the confidence levels overlap and their distributions are different (the lengths of the confidence interval bars are unequal). One of the solutions the authors propose for this problem (Figure 2c) is to allow users to indicate the y-axis value of interest (e.g. 42,000) and then draw a horizontal line and color bars based on this value. So bars might be colored red if they are definitely above this value (given the confidence interval), blue if they are definitely below this value, or white if they contain this value.
# 
# 
# <br>
# <img src="readonly/Assignment3Fig2c.png" alt="Figure 1" style="width: 400px;"/>
# <h4 style="text-align: center;" markdown="1">  Figure 2c from (Ferreira et al. 2014). Note that the colorbar legend at the bottom as well as the arrows are not required in the assignment descriptions below.</h4>
# 
# <br>
# <br>
# 
# **Easiest option:** Implement the bar coloring as described above - a color scale with only three colors, (e.g. blue, white, and red). Assume the user provides the y axis value of interest as a parameter or variable.
# 
# 
# **Harder option:** Implement the bar coloring as described in the paper, where the color of the bar is actually based on the amount of data covered (e.g. a gradient ranging from dark blue for the distribution being certainly below this y-axis, to white if the value is certainly contained, to dark red if the value is certainly not contained as the distribution is above the axis).
# 
# **Even Harder option:** Add interactivity to the above, which allows the user to click on the y axis to set the value of interest. The bar colors should change with respect to what value the user has selected.
# 
# **Hardest option:** Allow the user to interactively set a range of y values they are interested in, and recolor based on this (e.g. a y-axis band, see the paper for more details).
# 
# ---
# 
# *Note: The data given for this assignment is not the same as the data used in the article and as a result the visualizations may look a little different.*

# In[1]:


# Use the following data for this assignment:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.cm as cm

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                  index=[1992,1993,1994,1995])
df


# In[2]:


df_T = df.transpose()


# In[3]:


df_T.describe()


# In[4]:


import math
#yerr is calculated by std_sample/sqrt(Number of samples)
std_sample = df_T.describe().loc['std'].as_matrix()
sqrt_num_samples = np.sqrt( df_T.describe().loc['count'].astype(int).as_matrix())
yerr_dis = 1.96 * (std_sample/sqrt_num_samples)
yerr_dis


# In[7]:


def color_range(mean,err,line_pos):
    color_map = []
    for i in range(len(mean)):
        
        if  line_pos > (mean[i]+err[i]) :
            
            color_map.append('#0000ff')
        elif (mean[i]-err[i])<line_pos and (mean[i]+err[i])>line_pos:
            
            color_map.append('#b3b3b3')
        else:
            color_map.append('#ff0000')
            
    return color_map

def color_percentage(mean,err,line_pos):
    color_map = []
    for i in range(len(mean)):

        if line_pos > (mean[i]+err[i])  :
            color_map.append(0)
            
        elif (mean[i]-err[i])<line_pos and (mean[i]+err[i])>line_pos:
            
            color_map.append((1-(line_pos-(mean[i]-err[i]))/(2*err[i])))
        else:
            color_map.append(1)
        print(color_map)
    return color_map


# In[6]:


df_T.describe().loc['mean']
yerr_dis


# In[ ]:


line_pos = float(input(' Please Provide the y value  :'))
plt.figure()

objects = df_T.columns.astype(str)
y_pos = np.arange(len(objects))
performance = df_T.describe().loc['mean'].tolist()
color_map = color_range(df_T.describe().loc['mean'].tolist(),yerr_dis,line_pos)

plt.bar(y_pos, performance, align='center', alpha=0.5, yerr=yerr_dis, color=color_map)
plt.xticks(y_pos, objects) 
plt.ylabel('Usage')
plt.title('Programming language usage')
#create vertical line
plt.axhline(y=line_pos, color = 'b')

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# In[12]:


color_percentage(df_T.describe().loc['mean'].tolist(),yerr_dis,44000)


# In[22]:


from matplotlib.cm import ScalarMappable
#line_pos = float(input(' Please Provide the y value'))
line_pos=40000
fig, ax = plt.subplots(figsize=(15,10))

#plt.figure(figsize=(20,15))

color_map = color_percentage(df_T.describe().loc['mean'].tolist(),yerr_dis,line_pos)
color_map = cm.coolwarm(color_map)
#removing , alpha=0.5
rects = ax.bar(y_pos, performance, align='center', yerr=yerr_dis, color=color_map)
plt.xticks(y_pos, objects) 
plt.ylabel('Usage')
plt.title('Programming language usage')
#create vertical line
ax.axhline(y=line_pos,color = '#000000',alpha = 0.5, label = str(line_pos))
ax.text(-0.2, 1.03 * line_pos, 'line_pos  = '+str(line_pos), fontsize = 10 )
 
sm = ScalarMappable(cmap= cm.coolwarm, norm=plt.Normalize(0,1))
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Color', rotation=270,labelpad=25)
#plt.fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig('Assignment 3')


# In[14]:


color_percentage(df_T.describe().loc['mean'].tolist(),yerr_dis,line_pos)


# In[15]:




data_x = [0,1,2,3]
data_hight = [60,60,80,100]
data_color = [1000.,500.,1000.,900.]


data_color = [x / max(data_color) for x in data_color]
fig, ax = plt.subplots()

my_cmap = plt.cm.get_cmap('GnBu')
colors = my_cmap(data_color)
rects = ax.bar(data_x, data_hight, color=colors)

sm = ScalarMappable(cmap=my_cmap, norm=plt.Normalize(0,max(data_color)))
sm.set_array([])

cbar = plt.colorbar(sm)
cbar.set_label('Color', rotation=270,labelpad=25)

plt.xticks(data_x)    
plt.ylabel("Y")

plt.show()


# In[16]:


df_T.describe().loc['mean'].tolist()


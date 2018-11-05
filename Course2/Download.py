
# coding: utf-8

# In[ ]:


#resplace by your file name

import os

# create an archive file with all assignments and
# data files could have zip,tar, .tar.gz, .tgz, extensions
# check http://linuxcommand.org/man_pages/tar1.html for more info
filename = "course2data.zip"
if os.path.exists(filename):
    get_ipython().system(' rm "$filename"')
filesToCompress = '*.* readonly/*'

get_ipython().system('tar -czf "$filename" $filesToCompress > /dev/null')
from IPython.display import HTML
link = '<a href="{0}" target = _blank>Click here to download {0}</a>'
HTML(link.format(filename))


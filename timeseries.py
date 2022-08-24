#!/usr/bin/env python
# coding: utf-8

# In[1]:


#time series analysis
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
# Seaborn for plotting and styling
import seaborn as sb
df = sb.load_dataset('tips')
print (df.head())
df['total_bill'] = pd.to_datetime(df['total_bill'])
df.index = df['total_bill']
del df['total_bill']
df.plot(figsize=(15, 6))
plt.show()


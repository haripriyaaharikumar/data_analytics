import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

# read data
df = pd.read_csv("iris.csv") 
print(df.head()) 

# Assuming 'df' is your DataFrame 
quality_counts = df['sepal.width'].value_counts() 
  
# Using Matplotlib to create a count plot 
plt.figure(figsize=(8, 6)) 
plt.bar(quality_counts.index, quality_counts, color='darpink') 
plt.title('Count Plot of Quality') 
plt.xlabel('Quality') 
plt.ylabel('Count') 
plt.show() 

# Set Seaborn style 
sns.set_style("darkgrid") 
  
# Identify numerical columns 
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns 
  
# Plot distribution of each numerical feature 
plt.figure(figsize=(14, len(numerical_columns) * 3)) 
for idx, feature in enumerate(numerical_columns, 1): 
    plt.subplot(len(numerical_columns), 2, idx) 
    sns.histplot(df[feature], kde=True) 
    plt.title(f"{feature} | Skewness: {round(df[feature].skew(), 2)}") 
  
# Adjust layout and show plots 
plt.tight_layout() 
plt.show() 


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: Code1.ipynb
Conversion Date: 2025-10-31T19:14:06.931Z
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

df_fatlevel_male = pd.read_csv(r'E:\AUC\Fundamentals of DSCI II\Project\copy of cholesterol_fat_in_blood_men_mmolperl.csv')

df_lifeexpectancy_male= pd.read_csv(r'E:\AUC\Fundamentals of DSCI II\Project\life_expectancy_male_new.csv')


x1 =df_fatlevel_male.iloc[6,:29]
y1 =df_lifeexpectancy_male.iloc[6,:29]

x2 =np.array(df_fatlevel_male.iloc[10,:29])
y2 =np.array(df_lifeexpectancy_male.iloc[10,:29])

x3 =np.array(df_fatlevel_male.iloc[18,:29])
y3 =np.array(df_lifeexpectancy_male.iloc[18,:29])

x4 =np.array(df_fatlevel_male.iloc[25,:29])
y4 =np.array(df_lifeexpectancy_male.iloc[25,:29])

x5 =np.array(df_fatlevel_male.iloc[31,:29])
y5 =np.array(df_lifeexpectancy_male.iloc[31,:29])

x6 =np.array(df_fatlevel_male.iloc[39,:29])
y6 =np.array(df_lifeexpectancy_male.iloc[39,:29])

x7 =np.array(df_fatlevel_male.iloc[46,:29])
y7 =np.array(df_lifeexpectancy_male.iloc[46,:29])

x8 =np.array(df_fatlevel_male.iloc[54,:29])
y8 =np.array(df_lifeexpectancy_male.iloc[54,:29])


plt.scatter(x1,y1,color='orange', label = 'South America')
plt.scatter(x2,y2, color='green', label = 'North America')
plt.scatter(x3,y3,color='blue', label = 'Western Europe')
plt.scatter(x4,y4, color='black', label = 'Arab World')
plt.scatter(x5,y5,color='cyan', label = 'Central Africa')
plt.scatter(x6,y6, color='red', label = 'Central America')
plt.scatter(x7,y7,color='yellow', label = 'East Asia')
plt.scatter(x8,y8, color='purple', label = 'Nordic Countries')

plt.title('Male Fat in blood/Cholestrol Level vs Male Life Expectancy from 1980-2010') #title
plt.xlabel('Male Fat in blood/Cholestrol Level') #x label
plt.ylabel('Male Life Expectancy (Years)') #y label
#plt.xlim(left=0)
#plt.ylim(bottom=0)
plt.yticks(np.arange(30,90,10))
plt.xticks(np.arange(4,7,0.5))
plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

df_fatlevel_male = pd.read_csv(r'E:\AUC\Fundamentals of DSCI II\Project\copy of cholesterol_fat_in_blood_men_mmolperl.csv')

df_lifeexpectancy_male= pd.read_csv(r'E:\AUC\Fundamentals of DSCI II\Project\life_expectancy_male_new.csv')


x1 =df_fatlevel_male.iloc[5,1:29]
y1 =df_lifeexpectancy_male.iloc[5,1:29]

x2 =np.array(df_fatlevel_male.iloc[10,1:29])
y2 =np.array(df_lifeexpectancy_male.iloc[10,1:29])

x3 =np.array(df_fatlevel_male.iloc[18,1:29])
y3 =np.array(df_lifeexpectancy_male.iloc[18,1:29])

x4 =np.array(df_fatlevel_male.iloc[25,1:29])
y4 =np.array(df_lifeexpectancy_male.iloc[25,1:29])

x5 =np.array(df_fatlevel_male.iloc[31,1:29])
y5 =np.array(df_lifeexpectancy_male.iloc[31,1:29])

x6 =np.array(df_fatlevel_male.iloc[39,1:29])
y6 =np.array(df_lifeexpectancy_male.iloc[39,1:29])

x7 =np.array(df_fatlevel_male.iloc[46,1:29])
y7 =np.array(df_lifeexpectancy_male.iloc[46,1:29])

x8 =np.array(df_fatlevel_male.iloc[54,1:29])
y8 =np.array(df_lifeexpectancy_male.iloc[54,1:29])


covariance_matrix_1 = np.cov(x1.astype(float), y1.astype(float))
covariance_matrix_2 = np.cov(x2.astype(float), y2.astype(float))
covariance_matrix_3 = np.cov(x3.astype(float), y3.astype(float))
covariance_matrix_4 = np.cov(x4.astype(float), y4.astype(float))
covariance_matrix_5 = np.cov(x5.astype(float), y5.astype(float))
covariance_matrix_6 = np.cov(x6.astype(float), y6.astype(float))
covariance_matrix_7 = np.cov(x7.astype(float), y7.astype(float))
covariance_matrix_8 = np.cov(x8.astype(float), y8.astype(float))


# The covariance is the element in the 0th row and 1st column (or vice versa)
South_America_covariance = covariance_matrix_1[0, 1]

North_America_covariance = covariance_matrix_2[0, 1]

Western_Europe_covariance = covariance_matrix_3[0, 1]

Arab_World_covariance = covariance_matrix_4[0, 1]

Central_Africa_covariance = covariance_matrix_5[0, 1]

Central_America_covariance = covariance_matrix_6[0, 1]

East_Asia_covariance = covariance_matrix_7[0, 1]

Nordic_Countries_covariance = covariance_matrix_8[0, 1]

print(South_America_covariance)
print(North_America_covariance)
print(Western_Europe_covariance)
print(Arab_World_covariance)
print(Central_Africa_covariance)
print(Central_America_covariance)
print(East_Asia_covariance)
print(Nordic_Countries_covariance)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

df_fatlevel_female = pd.read_csv(r'E:\AUC\Fundamentals of DSCI II\Project\cholesterol_fat_in_blood_women_mmolperl.csv')
df_lifeexpectancy_female= pd.read_csv(r'E:\AUC\Fundamentals of DSCI II\Project/life_expectancy_female_new.csv')

x1 =df_fatlevel_female.iloc[6,:29]
y1 =df_lifeexpectancy_female.iloc[6,:29]

x2 =np.array(df_fatlevel_female.iloc[10,:29])
y2 =np.array(df_lifeexpectancy_female.iloc[10,:29])

x3 =np.array(df_fatlevel_female.iloc[18,:29])
y3 =np.array(df_lifeexpectancy_female.iloc[18,:29])

x4 =np.array(df_fatlevel_female.iloc[25,:29])
y4 =np.array(df_lifeexpectancy_female.iloc[25,:29])

x5 =np.array(df_fatlevel_female.iloc[31,:29])
y5 =np.array(df_lifeexpectancy_female.iloc[31,:29])

x6 =np.array(df_fatlevel_female.iloc[38,1:29])
y6 =np.array(df_lifeexpectancy_female.iloc[38,1:29])

x7 =np.array(df_fatlevel_female.iloc[46,:29])
y7 =np.array(df_lifeexpectancy_female.iloc[46,:29])

x8 =np.array(df_fatlevel_female.iloc[54,:29])
y8 =np.array(df_lifeexpectancy_female.iloc[54,:29])


plt.scatter(x1,y1,color='orange', label = 'South America')
plt.scatter(x2,y2, color='green', label = 'North America')
plt.scatter(x3,y3,color='blue', label = 'Western Europe')
plt.scatter(x4,y4, color='black', label = 'Arab World')
plt.scatter(x5,y5,color='cyan', label = 'Central Africa')
plt.scatter(x6,y6, color='red', label = 'Central America')
plt.scatter(x7,y7,color='yellow', label = 'East Asia')
plt.scatter(x8,y8, color='purple', label = 'Nordic Countries')

plt.title('Female Fat in blood/Cholestrol Level vs Female Life Expectancy (Years) from 1980-2010') #title
plt.ylabel('Female Life Expectancy (Years)') #x label
plt.xlabel('Female Fat in blood/Cholestrol Level') #y label
#plt.xlim(left=0)
#plt.ylim(bottom=0)
plt.yticks(np.arange(30,90,10))
plt.xticks(np.arange(4,7,0.5))
plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

df_fatlevel_female = pd.read_csv(r'E:\AUC\Fundamentals of DSCI II\Project\cholesterol_fat_in_blood_women_mmolperl.csv')
df_lifeexpectancy_male= pd.read_csv(r'E:\AUC\Fundamentals of DSCI II\Project/life_expectancy_female_new.csv')

x1 =df_fatlevel_female.iloc[6,1:29]
y1 =df_lifeexpectancy_male.iloc[6,1:29]

x2 =np.array(df_fatlevel_female.iloc[10,1:29])
y2 =np.array(df_lifeexpectancy_male.iloc[10,1:29])

x3 =np.array(df_fatlevel_female.iloc[18,1:29])
y3 =np.array(df_lifeexpectancy_male.iloc[18,1:29])

x4 =np.array(df_fatlevel_female.iloc[25,1:29])
y4 =np.array(df_lifeexpectancy_male.iloc[25,1:29])

x5 =np.array(df_fatlevel_female.iloc[31,1:29])
y5 =np.array(df_lifeexpectancy_male.iloc[31,1:29])

x6 =np.array(df_fatlevel_female.iloc[38,1:29])
y6 =np.array(df_lifeexpectancy_male.iloc[38,1:29])

x7 =np.array(df_fatlevel_female.iloc[46,1:29])
y7 =np.array(df_lifeexpectancy_male.iloc[46,1:29])

x8 =np.array(df_fatlevel_female.iloc[54,1:29])
y8 =np.array(df_lifeexpectancy_male.iloc[54,1:29])


covariance_matrix_1 = np.cov(x1.astype(float), y1.astype(float))
covariance_matrix_2 = np.cov(x2.astype(float), y2.astype(float))
covariance_matrix_3 = np.cov(x3.astype(float), y3.astype(float))
covariance_matrix_4 = np.cov(x4.astype(float), y4.astype(float))
covariance_matrix_5 = np.cov(x5.astype(float), y5.astype(float))
covariance_matrix_6 = np.cov(x6.astype(float), y6.astype(float))
covariance_matrix_7 = np.cov(x7.astype(float), y7.astype(float))
covariance_matrix_8 = np.cov(x8.astype(float), y8.astype(float))


# The covariance is the element in the 0th row and 1st column (or vice versa)
South_America_covariance = covariance_matrix_1[0, 1]

North_America_covariance = covariance_matrix_2[0, 1]

Western_Europe_covariance = covariance_matrix_3[0, 1]

Arab_World_covariance = covariance_matrix_4[0, 1]

Central_Africa_covariance = covariance_matrix_5[0, 1]

Central_America_covariance = covariance_matrix_6[0, 1]

East_Asia_covariance = covariance_matrix_7[0, 1]

Nordic_Countries_covariance = covariance_matrix_8[0, 1]

print(South_America_covariance)
print(North_America_covariance)
print(Western_Europe_covariance)
print(Arab_World_covariance)
print(Central_Africa_covariance)
print(Central_America_covariance)
print(East_Asia_covariance)
print(Nordic_Countries_covariance)

df_population_total.describe()

df_fatlevel_male.describe()

df_fatlevel_female.describe()

df_bmi_male.describe()

df_bmi_female.describe()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import os

df_lifeexpectancy_male= pd.read_csv(r'E:\AUC\Fundamentals of DSCI II\Project\life_expectancy_male_new.csv')
df_lifeexpectancy_female= pd.read_csv(r'E:\AUC\Fundamentals of DSCI II\Project\life_expectancy_female_new.csv')
df_fishconsumption= pd.read_csv(r'E:\AUC\Fundamentals of DSCI II\Project\fish_food_consmption_new.csv')


df_lifeexpectancy_male.describe()

df_lifeexpectancy_female.describe()

df_fishconsumption.describe()

import pandas as pd
import matplotlib.pyplot as plt

def create_histogram(csv_file_path, column_name):
    # Read data from CSV file using pandas
    data = pd.read_csv(csv_file_path)[column_name]

    # Create histogram
    plt.hist(data, bins=20, edgecolor='black')
    plt.xlabel('Fish Consumption per capita (kg)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Average Fish Consumption per region'.format(column_name))
    plt.show()

csv_file_path = 'E:\AUC\Fundamentals of DSCI II\Project\Averages.csv'
column_name = 'fish consumption'

create_histogram(csv_file_path, column_name)

df_fatlevel_female.iloc[39,:29]
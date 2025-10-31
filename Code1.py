#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-10-31T19:13:04.987Z
"""

#Fish Food vs Life expectancy Males Scatter Plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

df_fish = pd.read_csv(r'D:\DSCI 2\ready data\fish_food_consmption_new.csv')

df_life_expectancy_male= pd.read_csv(r'D:\DSCI 2\ready data\life_expectancy_male_new.csv')

x1 =np.array(df_fish.iloc[6,1:29])
y1 =np.array(df_life_expectancy_male.iloc[6,1:29])

x2 =np.array(df_fish.iloc[10,1:29])
y2 =np.array(df_life_expectancy_male.iloc[10,1:29])

x3 =np.array(df_fish.iloc[18,1:29])
y3 =np.array(df_life_expectancy_male.iloc[18,1:29])

x4 =np.array(df_fish.iloc[25,1:29])
y4 =np.array(df_life_expectancy_male.iloc[25,1:29])

x5 =np.array(df_fish.iloc[31,1:29])
y5 =np.array(df_life_expectancy_male.iloc[31,1:29])

x6 =np.array(df_fish.iloc[39,1:29])
y6 =np.array(df_life_expectancy_male.iloc[39,1:29])

x7 =np.array(df_fish.iloc[46,1:29])
y7 =np.array(df_life_expectancy_male.iloc[46,1:29])


x8 =np.array(df_fish.iloc[54,1:29])
y8 =np.array(df_life_expectancy_male.iloc[54,1:29])

def scatter(a , b , c , d):
    plt.scatter(a,b,color=c, label = d)
    
South_America_graph = scatter(x1 , y1 , 'orange' , 'South America' )
North_America_graph = scatter(x2 , y2 , 'green' , 'North America' )
Western_Europe_graph = scatter(x3 , y3 , 'blue' , 'Western Europe' )
Arab_World_graph = scatter(x4 , y4 , 'black' , 'Arab World' ) 
Central_Africa_graph = scatter(x5 , y5 , 'cyan' , 'Central Africa' )
Central_America_graph = scatter(x6 , y6 , 'red' , 'Central America' )
East_Asia_graph = scatter(x7 , y7 , 'yellow' , 'East Asia' )
Nordic_Countries_graph = scatter(x8 , y8 , 'purple' , 'Nordic Countries' )

plt.title('Fish Consumption vs Female Life Expectancy in different regions from 1980-2008') #title
plt.xlabel('Fish and seafood consumption per capita per year (kg)') #x label
plt.ylabel('Life Expectancy (Years)') #y label
plt.xticks(np.arange(0, 40, 2))
plt.yticks(np.arange(30, 100, 5))
plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()










#Fish Food vs Life Expectancy Females Scatter Plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
df_fish = pd.read_csv(r'D:\DSCI 2\ready data\fish_food_consmption_new.csv')
df_life_expectancy_female= pd.read_csv(r'D:\DSCI 2\ready data\life_expectancy_female_new.csv')

x1 =np.array(df_fish.iloc[6,1:29])
y1 =np.array(df_life_expectancy_female.iloc[6,1:29])
x2 =np.array(df_fish.iloc[10,1:29])
y2 =np.array(df_life_expectancy_female.iloc[10,1:29])
x3 =np.array(df_fish.iloc[18,1:29])
y3 =np.array(df_life_expectancy_female.iloc[18,1:29])
x4 =np.array(df_fish.iloc[25,1:29])
y4 =np.array(df_life_expectancy_female.iloc[25,1:29])
x5 =np.array(df_fish.iloc[31,1:29])
y5 =np.array(df_life_expectancy_female.iloc[31,1:29])
x6 =np.array(df_fish.iloc[39,1:29])
y6 =np.array(df_life_expectancy_female.iloc[38,1:29])
x7 =np.array(df_fish.iloc[46,1:29])
y7 =np.array(df_life_expectancy_female.iloc[46,1:29])
x8 =np.array(df_fish.iloc[54,1:29])
y8 =np.array(df_life_expectancy_female.iloc[54,1:29])

def scatter(a , b , c , d):
    plt.scatter(a,b,color=c, label = d)
South_America_graph = scatter(x1 , y1 , 'orange' , 'South America' )
North_America_graph = scatter(x2 , y2 , 'green' , 'North America' )
Western_Europe_graph = scatter(x3 , y3 , 'blue' , 'Western Europe' )
Arab_World_graph = scatter(x4 , y4 , 'black' , 'Arab World' ) 
Central_Africa_graph = scatter(x5 , y5 , 'cyan' , 'Central Africa' )
Central_America_graph = scatter(x6 , y6 , 'red' , 'Central America' )
East_Asia_graph = scatter(x7 , y7 , 'yellow' , 'East Asia' )
Nordic_Countries_graph = scatter(x8 , y8 , 'purple' , 'Nordic Countries' )

plt.title('Fish Consumption vs Female Life Expectancy in different regions from 1980-2008') #title
plt.xlabel('Fish and seafood consumption per capita per year (kg)') #x label
plt.ylabel('Life Expectancy (Years)') #y label
plt.xticks(np.arange(0, 40, 2))
plt.yticks(np.arange(30, 100, 5))
plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#Fish Food vs Life expectancy Males Covariance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

df_fish = pd.read_csv(r'D:\DSCI 2\ready data\fish_food_consmption_new.csv')

df_life_expectancy_male= pd.read_csv(r'D:\DSCI 2\ready data\life_expectancy_male_new.csv')


x1 =np.array(df_fish.iloc[6,1:29])
y1 =np.array(df_life_expectancy_male.iloc[6,1:29])

x2 =np.array(df_fish.iloc[10,1:29])
y2 =np.array(df_life_expectancy_male.iloc[10,1:29])

x3 =np.array(df_fish.iloc[18,1:29])
y3 =np.array(df_life_expectancy_male.iloc[18,1:29])

x4 =np.array(df_fish.iloc[25,1:29])
y4 =np.array(df_life_expectancy_male.iloc[25,1:29])

x5 =np.array(df_fish.iloc[31,1:29])
y5 =np.array(df_life_expectancy_male.iloc[31,1:29])

x6 =np.array(df_fish.iloc[39,1:29])
y6 =np.array(df_life_expectancy_male.iloc[39,1:29])

x7 =np.array(df_fish.iloc[46,1:29])
y7 =np.array(df_life_expectancy_male.iloc[46,1:29])


x8 = np.array(df_fish.iloc[54, 1:29])
y8 = np.array(df_life_expectancy_male.iloc[54, 1:29])

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




#Fish Food vs Life expectancy Females Covariance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

df_fish = pd.read_csv(r'D:\DSCI 2\ready data\fish_food_consmption_new.csv')

df_life_expectancy_female= pd.read_csv(r'D:\DSCI 2\ready data\life_expectancy_female_new.csv')


x1 =np.array(df_fish.iloc[6,1:29])
y1 =np.array(df_life_expectancy_female.iloc[6,1:29])

x2 =np.array(df_fish.iloc[10,1:29])
y2 =np.array(df_life_expectancy_female.iloc[10,1:29])

x3 =np.array(df_fish.iloc[18,1:29])
y3 =np.array(df_life_expectancy_female.iloc[18,1:29])

x4 =np.array(df_fish.iloc[25,1:29])
y4 =np.array(df_life_expectancy_female.iloc[25,1:29])

x5 =np.array(df_fish.iloc[31,1:29])
y5 =np.array(df_life_expectancy_female.iloc[31,1:29])

x6 =np.array(df_fish.iloc[39,1:29])
y6 =np.array(df_life_expectancy_female.iloc[38,1:29])

x7 =np.array(df_fish.iloc[46,1:29])
y7 =np.array(df_life_expectancy_female.iloc[46,1:29])


x8 = np.array(df_fish.iloc[54, 1:29])
y8 = np.array(df_life_expectancy_female.iloc[54, 1:29])

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













#Regression: Supervised Learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from sklearn.linear_model import LinearRegression

df_lr_fish =pd.read_csv(r'D:\DSCI 2\Project\Machine Learning\Fish Machine Learning.csv')
df_lr_life_exp_male =pd.read_csv(r'D:\DSCI 2\Project\Machine Learning\life exp males machine learning.csv')

x_train = np.array(df_lr_fish.iloc[41,10:20])  
y_train = np.array(df_lr_life_exp_male.iloc[41,10:20])

lr = LinearRegression()

x_train_new = x_train.reshape(1, -1)

y_train_new = y_train.reshape(1, -1)

lrModel = lr.fit(x_train_new , y_train_new)

x_test = np.array(df_lr_fish.iloc[41,20:30])

x_test_new = x_test.reshape(1, -1)

x=x_test_new.flatten()

y_actual = np.array(df_lr_life_exp_male.iloc[41,20:30])

y_actual_new = y_actual.reshape(1, -1)

y=y_actual_new.flatten()

y_pred = lrModel.predict(x_test_new)

y_reg = y_pred.flatten()

plt.scatter(x.astype('float64') , y.astype('float64'), label=' Dataset with Labeled X and Y')
sns.regplot(x.astype('float64') , y.astype('float64'))

plt.scatter(x.astype('float64'), y_pred, label=' Dataset with Labeled X and Hidden Y')
sns.regplot(x.astype('float64'), y_reg.astype('float64'))

plt.title('Actual and Prediction of Average Fish Consumption vs Average Male Life Expectancy of all regions from 2009-2018') #title
plt.xlabel('Average Fish and seafood consumption per capita per year (kg) of all regions') #x label
plt.ylabel('Average Life Expectancy (Years)') #y label


plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
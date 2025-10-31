#!/usr/bin/env python
# coding: utf-8

# In[288]:


#Project Code file: First step: importing packages and reading each seperate excel sheet as its own data frame.

import seaborn as sbn
import pandas as pd 
import matplotlib.pyplot as plt
import os
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[289]:


os.chdir("D:\AUC\Python Data\Project Data\Dataset 1 Project")


# In[290]:


os.getcwd()


# In[291]:


malexp = pd.read_csv("life_expectancy_male_new(1).csv")


# In[292]:


femalexp = pd.read_csv("life_expectancy_female_new(1).csv")


# In[293]:


chlstrlmen = pd.read_csv("Chlstrl Men.csv")


# In[294]:


chlstrlwomen = pd.read_csv("Chlstrl Women.csv")


# In[295]:


BMImen = pd.read_csv("BMI Men.csv")


# In[296]:


BMIwomen = pd.read_csv("BMIWomen1.csv")


# In[297]:


malexp.head()


# In[298]:


malexp2 = malexp.copy()


# In[299]:


malexp_new=malexp2.drop(["2009","2010"],axis = 1)


# In[300]:


malexp_new.dropna()


# In[301]:


malexp_new.head()


# In[302]:


femalexp.head()


# In[303]:


femalexp2 = femalexp.copy()


# In[304]:


femalexp2.dropna()


# In[305]:


femalexp_new = femalexp2.drop(["2009","2010"],axis =1)


# In[306]:


femalexp_new.head()


# In[307]:


chlstrlmen.head()


# In[308]:


chlstrlmen.dropna()


# In[309]:


chlstrlwomen.head()


# In[310]:


chlstrlwomen.dropna()


# In[311]:


BMImen.head()


# In[312]:


BMImen.dropna()


# In[313]:


BMIwomen.head()


# In[314]:


BMIwomen.dropna()


# In[315]:


malexp2.describe()


# In[316]:


femalexp2.describe()


# In[320]:


BMImen.describe()


# In[321]:


BMIwomen.describe()


# In[322]:


chlstrlmen.describe()


# In[323]:


chlstrlwomen.describe()


# In[324]:


#The following section revolves around analyzing the relationship between BMI & Life Expectancy for both genders regionally.


# In[325]:


# ------------------------------Data Analysis for BMI Men-------------------------------------


# In[326]:


SA_BMI_men = BMImen.iloc[0:6, 1:30]
SAex_men = malexp_new.iloc[0:6, 1:30]
NA_BMI_men = BMImen.iloc[8:10, 1:30]
NAex_men = malexp_new.iloc[8:10, 1:30]
WE_BMI_men = BMImen.iloc[12:18, 1:30]
WEex_men = malexp_new.iloc[12:18, 1:30]
AW_BMI_men = BMImen.iloc[20:25, 1:30]
AWex_men = malexp_new.iloc[20:25, 1:30]
CA_BMI_men = BMImen.iloc[27:31, 1:30]
CAex_men = malexp_new.iloc[27:31, 1:30]
CAM_BMI_men = BMImen.iloc[33:39, 1:30]
CAMex_men = malexp_new.iloc[33:39, 1:30]
EA_BMI_men = BMImen.iloc[41:46, 1:30]
EAex_men = malexp_new.iloc[41:46, 1:30]
NC_BMI_men = BMImen.iloc[48:54, 1:30] 
NCex_men = malexp_new.iloc[48:54, 1:30]


# In[327]:


SA_BMI_men.head()


# In[328]:


meanSABMI = SA_BMI_men.mean(axis = 0)
meanSAex = SAex_men.mean(axis = 0)
meanNABMI = NA_BMI_men.mean(axis = 0)
meanNAex = NAex_men.mean(axis = 0)
meanWEBMI = WE_BMI_men.mean(axis = 0)
meanWEex = WEex_men.mean(axis = 0)
meanAWBMI = AW_BMI_men.mean(axis = 0)
meanAWex = AWex_men.mean(axis = 0)
meanCABMI = CA_BMI_men.mean(axis = 0)
meanCAex = CAex_men.mean(axis = 0)
meanCAMBMI = CAM_BMI_men.mean(axis = 0)
meanCAMex = CAMex_men.mean(axis = 0)
meanEABMI = EA_BMI_men.mean(axis = 0)
meanEAex = EAex_men.mean(axis = 0)
meanNCBMI = NC_BMI_men.mean(axis = 0)
meanNCex = NCex_men.mean(axis = 0)


# In[329]:



meanSABMI[28]


# In[330]:


exdf = {"1980":[meanSAex[1],meanNAex[1],meanWEex[1],meanAWex[1],meanCAex[1],meanCAMex[1],meanEAex[1],meanNCex[1]],
        "1990":[meanSAex[10],meanNAex[10],meanWEex[10],meanAWex[10],meanCAex[10],meanCAMex[10],meanEAex[10],meanNCex[10]],
        "2000":[meanSAex[20],meanNAex[20],meanWEex[20],meanAWex[20],meanCAex[20],meanCAMex[20],meanEAex[20],meanNCex[20]],
        "2008":[meanSAex[28],meanNAex[28],meanWEex[28],meanAWex[28],meanCAex[28],meanCAMex[28],meanEAex[28],meanNCex[28]]}


# In[331]:


exdf = pd.DataFrame(exdf)
exdf.index=["SA","NA","WE","AW","CA","CAM","EA","NC"]
exdf


# In[332]:


newdf = {"1980":[meanSABMI[1],meanNABMI[1],meanWEBMI[1],meanAWBMI[1],meanCABMI[1],meanCAMBMI[1],meanEABMI[1],meanNCBMI[1]],
        "1990":[meanSABMI[10],meanNABMI[10],meanWEBMI[10],meanAWBMI[10],meanCABMI[10],meanCAMBMI[10],meanEABMI[10],meanNCBMI[10]],
        "2000":[meanSABMI[20],meanNABMI[20],meanWEBMI[20],meanAWBMI[20],meanCABMI[20],meanCAMBMI[20],meanEABMI[20],meanNCBMI[20]],
        "2008":[meanSABMI[28],meanNABMI[28],meanWEBMI[28],meanAWBMI[28],meanCABMI[28],meanCAMBMI[28],meanEABMI[28],meanNCBMI[28]]}


# In[333]:


newdf = pd.DataFrame(newdf)
newdf.index=["SA","NA","WE","AW","CA","CAM","EA","NC"]
newdf


# In[334]:


z1=np.array(meanSABMI)
k1=np.array(meanSAex)
z2=np.array(meanNABMI)
k2=np.array(meanNAex)
z3=np.array(meanWEBMI)
k3=np.array(meanWEex)
z4=np.array(meanAWBMI)
k4=np.array(meanAWex)
z5=np.array(meanCABMI)
k5=np.array(meanCAex)
z6=np.array(meanCAMBMI)
k6=np.array(meanCAMex)
z7=np.array(meanEABMI)
k7=np.array(meanEAex)
z8=np.array(meanNCBMI)
k8=np.array(meanNCex)
Z = [z1,z2,z3,z4,z5,z6,z7,z8]
K = [k1,k2,k3,k4,k5,k6,k7,k8]


# In[335]:


plt.scatter(z1,k1, color='orange',s=20, label = 'South America')
plt.scatter(z2,k2, color='green',s=20, label = 'North America')
plt.scatter(z3,k3,color='blue',s = 20, label = 'Western Europe')
plt.scatter(z4,k4, color='pink',s=  20, label = 'Arab World')
plt.scatter(z5,k5,color='cyan',s= 20, label = 'Central Africa')
plt.scatter(z6,k6, color='red',s = 20, label = 'Central America')
plt.scatter(z7,k7,color='yellow',s = 20, label = 'East Asia')
plt.scatter(z8,k8, color='purple',s = 20, label = 'Nordic Countries')
plt.title('Male body mass index (BMI) vs male life expectancy in different regions from 1980-2008') #title
plt.xlabel("BMI (in kg/m^2)")
plt.ylabel("Life Expectancy (in years)")
plt.xticks(np.arange(18, 30, 2))
plt.yticks(np.arange(30, 85, 5))
plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[336]:


z1


# In[337]:


Cov_BMI = np.cov(z1,k1)


# In[338]:


print(Cov_BMI)


# In[339]:


# ----------------------------------------------Data Analysis for BMI Women--------------------------------------------


# In[340]:


SA_BMI_women = BMIwomen.iloc[0:6, 1:30]
SAex_women = femalexp_new.iloc[0:6, 1:30]
NA_BMI_women = BMIwomen.iloc[8:10, 1:30]
NAex_women = femalexp_new.iloc[8:10, 1:30]
WE_BMI_women = BMIwomen.iloc[12:18, 1:30]
WEex_women = femalexp_new.iloc[12:18, 1:30]
AW_BMI_women = BMIwomen.iloc[20:25, 1:30]
AWex_women = femalexp_new.iloc[20:25, 1:30]
CA_BMI_women = BMIwomen.iloc[27:31, 1:30]
CAex_women = femalexp_new.iloc[27:31, 1:30]
CAM_BMI_women = BMIwomen.iloc[33:39, 1:30]
CAMex_women = femalexp_new.iloc[33:39, 1:30]
EA_BMI_women = BMIwomen.iloc[41:46, 1:30]
EAex_women = femalexp_new.iloc[41:46, 1:30]
NC_BMI_women = BMIwomen.iloc[48:54, 1:30] 
NCex_women = femalexp_new.iloc[48:54, 1:30]


# In[341]:


meanSABMI_w = SA_BMI_women.mean(axis = 0)
meanSAex_w = SAex_women.mean(axis = 0)
meanNABMI_w = NA_BMI_women.mean(axis = 0)
meanNAex_w = NAex_women.mean(axis = 0)
meanWEBMI_w = WE_BMI_women.mean(axis = 0)
meanWEex_w = WEex_women.mean(axis = 0)
meanAWBMI_w = AW_BMI_women.mean(axis = 0)
meanAWex_w = AWex_women.mean(axis = 0)
meanCABMI_w = CA_BMI_women.mean(axis = 0)
meanCAex_w = CAex_women.mean(axis = 0)
meanCAMBMI_w = CAM_BMI_women.mean(axis = 0)
meanCAMex_w = CAMex_women.mean(axis = 0)
meanEABMI_w = EA_BMI_women.mean(axis = 0)
meanEAex_w = EAex_women.mean(axis = 0)
meanNCBMI_w = NC_BMI_women.mean(axis = 0)
meanNCex_w = NCex_women.mean(axis = 0)


# In[342]:


exdf_w = {"1980":[meanSAex_w[1],meanNAex_w[1],meanWEex_w[1],meanAWex_w[1],meanCAex_w[1],meanCAMex_w[1],meanEAex_w[1],meanNCex_w[1]],
        "1990":[meanSAex_w[10],meanNAex_w[10],meanWEex_w[10],meanAWex_w[10],meanCAex_w[10],meanCAMex_w[10],meanEAex_w[10],meanNCex_w[10]],
        "2000":[meanSAex_w[20],meanNAex_w[20],meanWEex_w[20],meanAWex_w[20],meanCAex_w[20],meanCAMex_w[20],meanEAex_w[20],meanNCex_w[20]],
        "2008":[meanSAex_w[28],meanNAex_w[28],meanWEex_w[28],meanAWex_w[28],meanCAex_w[28],meanCAMex_w[28],meanEAex_w[28],meanNCex_w[28]]}


# In[343]:


exdf_w = pd.DataFrame(exdf_w)
exdf_w.index=["SA","NA","WE","AW","CA","CAM","EA","NC"]
exdf_w


# In[344]:


newdf_w = {"1980":[meanSABMI_w[1],meanNABMI_w[1],meanWEBMI_w[1],meanAWBMI_w[1],meanCABMI_w[1],meanCAMBMI_w[1],meanEABMI_w[1],meanNCBMI_w[1]],
        "1990":[meanSABMI_w[10],meanNABMI_w[10],meanWEBMI_w[10],meanAWBMI_w[10],meanCABMI_w[10],meanCAMBMI_w[10],meanEABMI_w[10],meanNCBMI_w[10]],
        "2000":[meanSABMI_w[20],meanNABMI_w[20],meanWEBMI_w[20],meanAWBMI_w[20],meanCABMI_w[20],meanCAMBMI_w[20],meanEABMI_w[20],meanNCBMI_w[20]],
        "2008":[meanSABMI_w[28],meanNABMI_w[28],meanWEBMI_w[28],meanAWBMI_w[28],meanCABMI_w[28],meanCAMBMI_w[28],meanEABMI_w[28],meanNCBMI_w[28]]}


# In[345]:


newdf_w = pd.DataFrame(newdf_w)
newdf_w.index=["SA","NA","WE","AW","CA","CAM","EA","NC"]
newdf_w


# In[346]:


j1=np.array(meanSABMI_w)
i1=np.array(meanSAex_w)
j2=np.array(meanNABMI_w)
i2=np.array(meanNAex_w)
j3=np.array(meanWEBMI_w)
i3=np.array(meanWEex_w)
j4=np.array(meanAWBMI_w)
i4=np.array(meanAWex_w)
j5=np.array(meanCABMI_w)
i5=np.array(meanCAex_w)
j6=np.array(meanCAMBMI_w)
i6=np.array(meanCAMex_w)
j7=np.array(meanEABMI_w)
i7=np.array(meanEAex_w)
j8=np.array(meanNCBMI_w)
i8=np.array(meanNCex_w)
J = [j1,j2,j3,j4,j5,j6,j7,j8]
I = [i1,i2,i3,i4,i5,i6,i7,i8]


# In[347]:


plt.scatter(j1,i1,color='orange',s=20, label = 'South America')
plt.scatter(j2,i2, color='green',s=20, label = 'North America')
plt.scatter(j3,i3,color='blue',s = 20, label = 'Western Europe')
plt.scatter(j4,i4, color='pink',s=  20, label = 'Arab World')
plt.scatter(j5,i5,color='cyan',s= 20, label = 'Central Africa')
plt.scatter(j6,i6, color='red',s = 20, label = 'Central America')
plt.scatter(j7,i7,color='yellow',s = 20, label = 'East Asia')
plt.scatter(j8,i8, color='purple',s = 20, label = 'Nordic Countries')
plt.title('Female body mass index (BMI) vs female life expectancy in different regions from 1980-2008') #title
plt.xlabel('BMI (kg/m^2)') #x label
plt.ylabel('Life Expectancy (Years)') #y label

plt.xticks(np.arange(18, 30, 2))
plt.yticks(np.arange(30, 85, 5))
plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[348]:


#------------------Female mean BMI and Life expectancy (per region over the time period from 1980 to 2008)---------------


# In[349]:




meanfemale = {'SA female': [meanSABMI_w,meanSAex_w],
             'NA female': [meanNABMI_w,meanNAex_w],
             'WE female': [meanWEBMI_w,meanWEex_w],
             'AW female': [meanAWBMI_w,meanAWex_w],
             'CA female': [meanCABMI_w,meanCAex_w],
             'CAM female': [meanCAMBMI_w,meanCAMex_w],
             'EA female': [meanEABMI_w,meanEAex_w],
             'NC female': [meanNCBMI_w,meanNCex_w]}
print(meanfemale)


# In[350]:


meanfemale_df = pd.DataFrame(meanfemale)


# In[351]:


meanfemale_df.head()


# In[352]:


meanfemale_df.index=['BMI','Life Expectancy']


# In[353]:


print(meanfemale_df.head())


# In[354]:


#Buffer -------------------------------------------------------------------


# In[355]:



#BUFFER------------------------------------------------------------------


# In[356]:


#BUFFER--------------------------------------------------------------------


# In[357]:


import statsmodels.api as sm


# In[358]:


#NA BMI vs. Life Expectancy
xm1 = meanNABMI
ym1 = meanNAex
xm1 = sm.add_constant(xm1)
model = sm.OLS(ym1, xm1, missing='drop')
model_result = model.fit()
model_result.summary()


# In[359]:


sm.graphics.plot_fit(model_result,1, vlines = False)


# In[360]:


#SA BMI vs. Life Expectancy
xm2 = meanSABMI
ym2 = meanSAex
xm2 = sm.add_constant(xm2)
model2 = sm.OLS(ym2, xm2, missing='drop')
model_result2 = model2.fit()
model_result2.summary()


# In[361]:


sm.graphics.plot_fit(model_result2,1, vlines=False)


# In[362]:


#WE BMI vs. Life Expectancy
xm3 = meanWEBMI
ym3 = meanWEex
xm3 = sm.add_constant(xm3)
model3 = sm.OLS(ym3, xm3, missing='drop')
model_result3 = model3.fit()
model_result3.summary()


# In[363]:


sm.graphics.plot_fit(model_result3,1, vlines=False)


# In[364]:


#AW BMI vs. Life Expectancy
xm4 = meanAWBMI
ym4 = meanAWex
xm4 = sm.add_constant(xm4)
model4 = sm.OLS(ym4, xm4, missing='drop')
model_result4 = model4.fit()
model_result4.summary()


# In[365]:


sm.graphics.plot_fit(model_result4,1, vlines=False)


# In[366]:


#CA BMI vs. Life Expectancy
xm5 = meanCABMI
ym5 = meanCAex
xm5 = sm.add_constant(xm5)
model5 = sm.OLS(ym5, xm5, missing='drop')
model_result5 = model5.fit()
model_result5.summary()


# In[367]:


sm.graphics.plot_fit(model_result5,1, vlines=False)


# In[368]:


#CAM BMI vs. Life Expectancy
xm6 = meanCAMBMI
ym6 = meanCAMex
xm6 = sm.add_constant(xm6)
model6 = sm.OLS(ym6, xm6, missing='drop')
model_result6 = model6.fit()
model_result6.summary()


# In[369]:


sm.graphics.plot_fit(model_result6,1, vlines=False)


# In[370]:


#EA BMI vs. Life Expectancy
xm7 = meanEABMI
ym7 = meanEAex
xm7 = sm.add_constant(xm7)
model7 = sm.OLS(ym7, xm7, missing='drop')
model_result7 = model7.fit()
model_result7.summary()


# In[371]:


sm.graphics.plot_fit(model_result7,1, vlines=False)


# In[372]:


#EA BMI vs. Life Expectancy
xm8 = meanNCBMI
ym8 = meanNCex
xm8 = sm.add_constant(xm8)
model8 = sm.OLS(ym8, xm8, missing='drop')
model_result8 = model8.fit()
model_result8.summary()


# In[373]:


sm.graphics.plot_fit(model_result8,1, vlines=False)


# In[374]:


#NA BMI vs. Life Expectancy
xm1_w = meanNABMI_w
ym1_w = meanNAex_w
xm1_w = sm.add_constant(xm1_w)
model_w = sm.OLS(ym1_w, xm1_w, missing='drop')
model_result_w = model_w.fit()
model_result_w.summary()


# In[375]:


sm.graphics.plot_fit(model_result_w,1, vlines=False)


# In[376]:


#NA BMI vs. Life Expectancy
xm2_w = meanSABMI_w
ym2_w = meanSAex_w
xm2_w = sm.add_constant(xm2_w)
model2_w = sm.OLS(ym2_w, xm2_w, missing='drop')
model_result2_w = model2_w.fit()
model_result2_w.summary()


# In[377]:


sm.graphics.plot_fit(model_result2_w,1, vlines=False)


# In[378]:


#NA BMI vs. Life Expectancy
xm3_w = meanWEBMI_w
ym3_w = meanWEex_w
xm3_w = sm.add_constant(xm3_w)
model3_w = sm.OLS(ym3_w, xm3_w, missing='drop')
model_result3_w = model3_w.fit()
model_result3_w.summary()


# In[379]:


sm.graphics.plot_fit(model_result3_w,1, vlines=False)


# In[380]:


#NA BMI vs. Life Expectancy
xm4_w = meanAWBMI_w
ym4_w = meanAWex_w
xm4_w = sm.add_constant(xm4_w)
model4_w = sm.OLS(ym4_w, xm4_w, missing='drop')
model_result4_w = model4_w.fit()
model_result4_w.summary()


# In[381]:


sm.graphics.plot_fit(model_result4_w,1, vlines=False)


# In[382]:


#NA BMI vs. Life Expectancy
xm5_w = meanCABMI_w
ym5_w = meanCAex_w
xm5_w = sm.add_constant(xm5_w)
model5_w = sm.OLS(ym5_w, xm5_w, missing='drop')
model_result5_w = model5_w.fit()
model_result5_w.summary()


# In[383]:


sm.graphics.plot_fit(model_result5_w,1, vlines=False)


# In[384]:


#NA BMI vs. Life Expectancy
xm6_w = meanCAMBMI_w
ym6_w = meanCAMex_w
xm6_w = sm.add_constant(xm6_w)
model6_w = sm.OLS(ym6_w, xm6_w, missing='drop')
model_result6_w = model6_w.fit()
model_result6_w.summary()


# In[385]:


sm.graphics.plot_fit(model_result6_w,1, vlines=False)


# In[386]:


#NA BMI vs. Life Expectancy
xm7_w = meanEABMI_w
ym7_w = meanEAex_w
xm7_w = sm.add_constant(xm7_w)
model7_w = sm.OLS(ym7_w, xm7_w, missing='drop')
model_result7_w = model7_w.fit()
model_result7_w.summary()


# In[387]:


sm.graphics.plot_fit(model_result7_w,1, vlines=False)


# In[388]:


#NA BMI vs. Life Expectancy
xm8_w = meanNCBMI_w
ym8_w = meanNCex_w
xm8_w = sm.add_constant(xm8_w)
model8_w = sm.OLS(ym8_w, xm8_w, missing='drop')
model_result8_w = model8_w.fit()
model_result8_w.summary()


# In[390]:


sm.graphics.plot_fit(model_result8_w,1, vlines=False)


# In[ ]:





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


data=pd.read_csv('sleepdataset.csv')
data=data.drop(['Person ID'], axis=1)
data.rename(columns={'Sleep Disorder':'Sleep_Disorder'}, inplace=True)
data.rename(columns={'BMI Category':'BMI_Category'}, inplace=True)
data.rename(columns={'Blood Pressure':'Blood_Pressure'}, inplace=True)

x=data.drop(['Quality of Sleep'], axis=1)
y=data['Quality of Sleep']

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)
"""
                              Age  Sleep Duration  Quality of Sleep  Physical Activity Level  Stress Level  Heart Rate  Daily Steps  Insomnia      None  Sleep Apnea
Age                      1.000000        0.299253          0.438826                 0.216244     -0.393411   -0.199048     0.128068  0.079326 -0.449819     0.462693
Sleep Duration           0.299253        1.000000          0.872999                 0.256255     -0.801708   -0.510881     0.015065 -0.335471*  0.337962    -0.079956
Quality of Sleep         0.438826        0.872999          1.000000                 0.229575     -0.894099   -0.671400     0.072085 -0.326315*  0.315317    -0.061672
Physical Activity Level  0.216244        0.256255          0.229575                 1.000000     -0.070748    0.099224     0.797898 -0.321143* -0.068196     0.393422
Stress Level            -0.393411       -0.801708         -0.894099                -0.070748      1.000000    0.695791     0.132241  0.105393 -0.169996     0.101687
Heart Rate              -0.199048       -0.510881         -0.671400                 0.099224      0.695791    1.000000    -0.041871  0.044987 -0.342884     0.367718
Daily Steps              0.128068        0.015065          0.072085                 0.797898      0.132241   -0.041871     1.000000 -0.313337*  0.006925     0.295724
Insomnia                 0.079326       -0.335471         -0.326315                -0.321143      0.105393    0.044987    -0.313337  1.000000 -0.593095    -0.258748
None                    -0.449819        0.337962          0.315317                -0.068196     -0.169996   -0.342884     0.006925 -0.593095*  1.000000    -0.624252
Sleep Apnea              0.462693       -0.079956         -0.061672                 0.393422      0.101687    0.367718     0.295724 -0.258748* -0.624252     1.000000
"""


training_data=x_train.join(y_train)

training_data=training_data.join(pd.get_dummies(training_data.Sleep_Disorder)).drop(['Sleep_Disorder'], axis=1)
training_data=training_data.join(pd.get_dummies(training_data.BMI_Category)).drop(['BMI_Category'], axis=1)

training_data=training_data.drop(['Occupation'], axis=1)
training_data=training_data.drop(['Gender'], axis=1)
training_data=training_data.drop(['Blood_Pressure'], axis=1)



reg=LinearRegression()
scaler = StandardScaler()

x_train, y_train=training_data.drop(['Quality of Sleep'], axis=1), training_data['Quality of Sleep']
scaler = StandardScaler()
x_train_s=scaler.fit_transform(x_train)

reg.fit(x_train_s, y_train)

#dotad dane do trenowania, niżej test


testing_data=x_train.join(y_train)



scaler = StandardScaler()

x_test, y_test=testing_data.drop(['Quality of Sleep'], axis=1), testing_data['Quality of Sleep']
print(type(testing_data.describe(percentiles=[.01,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99])))
x_test_s=scaler.fit_transform(x_test)
print(x_test_s[0],x_test, reg.predict(x_test_s[0].reshape(1,-1)))
save_model=pickle.dump(reg, open('model.pkl', 'wb'))
standard_population=testing_data.describe(percentiles=[.01,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99])
standard_pop=pickle.dump(standard_population, open('saved_standards.pkl', 'wb'))
save_scaler=pickle.dump(scaler, open('scaler.pkl', 'wb'))



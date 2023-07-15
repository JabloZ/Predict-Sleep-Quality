import pickle
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
pickled_model = pickle.load(open('model.pkl', 'rb'))
# Age  Sleep Duration  Physical Activity Level  Stress Level  Heart Rate  Daily Steps  Insomnia  None  Sleep Apnea  Normal  Normal Weight  Obese  Overweight
pickled_scaler=pickle.load(open('scaler.pkl', 'rb'))
pickled_standard=pickle.load(open('saved_standards.pkl', 'rb'))
print(pickled_standard.loc[:,'Sleep Duration'])
print('W skali 1-10 jaka będzie jakość twojego snu?')

age=int(input('wpisz wiek: '))
sleep_duration=float(input('ile godzin dziennie śpisz średnio? Jeśli śpisz np. 7h i 30 minut, wpisz 7.5: '))
physical_activity=int(input('jak w skali od 0 do 100 oceniłbyś/abyś poziom swojej aktywności fizycznej?: '))
stress_level=int(input('jak w skali od 0 do 10 oceniłbyś/abyś swój poziom stresu?: '))
heart_rate=int(input('ile wynosi twoje tętno spoczynkowe?: '))
daily_steps=int(input('ile krokow robisz srednio dziennie?: '))
sleep_type=int(input('Czy chorujesz na któreś z tych?: \n1.Bezsenność \n2.Bezdech senny \n3.żadne z powyższych\n: '))
insomnia=0
non=0
apnea=0
if sleep_type==1:
    insomnia+=1
elif sleep_type==2:
    apnea+=1
else:
    non+=1

bmi=float(input('wpisz swoje bmi: '))
weight=''
normal=0
overweight=0
obese=0
if bmi<25:
    weight='normal'
    normal+=1

elif bmi>=25 and bmi<30:
    weight='overweight'
    overweight+=1
else:
    weight='obese'
    obese+=1
#Age,Sleep Duration,Quality of Sleep,Physical Activity Level,Stress Level,BMI Category,Blood Pressure,Heart Rate,Daily Steps,Sleep Disorder
#28, 5.9,                  4,              30,                  8,                Obese,                  85,         3000,    Sleep Apnea
commands=['Input your age','Input your ']
x_train=np.array([age, sleep_duration, physical_activity, stress_level, heart_rate, daily_steps, insomnia, non, apnea, normal, 0, obese, overweight]).reshape(1,-1)

x_train_s=pickled_scaler.transform(x_train)
print('Twoja jakość snu w skali od 1-10 najprawdopodobniej wynosi... ',math.floor(round(pickled_model.predict(x_train_s)[0],2)))
print(pickled_model.predict(x_train_s))
percent=''
for index, row in pickled_standard.iterrows():
    
    if sleep_duration>row['Sleep Duration']:
        percent=index
print(f'dodatkowe statystyki: \nŚpisz więcej niż {percent} ludzi')
percent_quality=0
round(pickled_model.predict(x_train_s)[0],2)
for index, row in pickled_standard.iterrows():
    
    if pickled_model.predict(x_train_s)[0]>row['Quality of Sleep']:
        percent_quality=index
print(f'dodatkowe statystyki: \nTwój wynik snu jest lepszy niż u {percent_quality}')

""" DANE NA PODSTAWIE 300 RESPONDENTÓW
             Age        Sleep Duration      Physical Activity Level   Stress Level  Heart Rate   Daily Steps    Insomnia        None  Sleep Apnea      Normal  Normal Weight       Obese  Overweight  Quality of Sleep

1%      28.000000        5.998000                30.000000          3.000000        65.000000   3496.000000    0.000000    0.000000     0.000000    0.000000       0.000000    0.000000    0.000000          5.000000
10%     31.000000        6.080000                30.000000          3.000000        65.000000   5000.000000    0.000000    0.000000     0.000000    0.000000       0.000000    0.000000    0.000000          6.000000
20%     33.000000        6.200000                40.000000          4.000000        68.000000   5000.000000    0.000000    0.000000     0.000000    0.000000       0.000000    0.000000    0.000000          6.000000
30%     37.000000        6.500000                45.000000          4.000000        68.000000   6000.000000    0.000000    0.000000     0.000000    0.000000       0.000000    0.000000    0.000000          6.000000
40%     38.000000        6.940000                55.000000          5.000000        68.000000   6000.000000    0.000000    1.000000     0.000000    0.000000       0.000000    0.000000    0.000000          7.000000
50%     42.000000        7.200000                60.000000          5.000000        70.000000   7000.000000    0.000000    1.000000     0.000000    1.000000       0.000000    0.000000    0.000000          7.000000
60%     43.000000        7.400000                60.000000          6.000000        70.000000   7460.000000    0.000000    1.000000     0.000000    1.000000       0.000000    0.000000    0.000000          8.000000
70%     45.000000        7.700000                75.000000          7.000000        72.000000   8000.000000    0.000000    1.000000     0.000000    1.000000       0.000000    0.000000    1.000000          8.000000
80%     50.000000        7.800000                80.000000          7.000000        72.000000   8000.000000    0.000000    1.000000     0.400000    1.000000       0.000000    0.000000    1.000000          8.000000
90%     54.000000        8.200000                90.000000          8.000000        75.000000  10000.000000    1.000000    1.000000     1.000000    1.000000       0.000000    0.000000    1.000000          9.000000
95%     57.100000        8.400000                90.000000          8.000000        77.000000  10000.000000    1.000000    1.000000     1.000000    1.000000       1.000000    0.000000    1.000000          9.000000
99%     59.000000        8.500000                90.000000          8.000000        84.000000  10000.000000    1.000000    1.000000     1.000000    1.000000       1.000000    1.000000    1.000000          9.000000"""

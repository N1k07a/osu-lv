from sklearn import datasets
from sklearn . model_selection import train_test_split #Potrebno za razdvajanje test i train skupa
import pandas as pd #Ucitavanje csv datoteke
import matplotlib.pyplot as plt #Crtanje
from sklearn . preprocessing import StandardScaler #Scaliranje standradizacijom
from sklearn . preprocessing import MinMaxScaler #Scaliranje min max
import sklearn . linear_model as lm #Potreban za linearno regresijski model
from sklearn import metrics #Sluzi kako bi mogli nmapraviti evaluaciju modela na temelju metrica
import numpy as np

data = pd.read_csv('data_C02_emission.csv')

#-----Zadatak za podijelu test i train skupa podataka 80%-20%-------

X = data[['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)']]

y = data['CO2 Emissions (g/km)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 2) 

#-----Prikaz ovisnosti CO2 emission i neke druge numercike velicine gdje je jedna plavom bojom oznacena, a druga crvenom----
plt.subplot(1, 2, 1)
plt.scatter(X_train['Engine Size (L)'], y_train, c='blue')
plt.title('Training skup')
plt.xlabel('Engine Size (L)')
plt.ylabel('CO2 emission')
plt.subplot(1, 2, 2)
plt.scatter(X_test['Engine Size (L)'], y_test, c='red')
plt.title('Test skup')
plt.xlabel('Engine Size (L)')
plt.ylabel('CO2 emission')
plt.grid()
plt.show()

#-----Standardizacija podataka za ucenje(Standardizacijom ali moze i MinMaxScaler), prikaz pomocu histograma-------
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
plt.grid()
plt.subplot(1,2,1)
plt.hist(X_train['Engine Size (L)'])
plt.title("Originalni Engine Size")
plt.subplot(1,2,2)
plt.hist(X_train_n[ : , 0 ])
plt.title("Standardizirani Engine Size")
plt.show()
#----Standardizacija testnog skupa se uvjek izvodi sa transform ne fit-transform
X_test_n = sc.transform(X_test)

#Izrada linearno regresijskog modela i ispis parametara(theta) u terminal
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
print(f"Ovo su parametri za svaku vrijednost koju smo mi upisali za trening {linearModel.coef_}") #To su parametri iz formule theta1,theata2, ...
print(f"Ovo je nulti parametar koji nam pokazuje odsjecak na y-osi {linearModel.intercept_}") #To je parametar iz formule theata0

#-----Procijena izlazne velicine na temelju ulaznih velicina za testiranje i prikaz odnosa pomocu dijagrama rasprsenosti
#-----Odnosi se prikazuju tako da se na x-os stavi stvarne velicine, a na y-os predictane i nacrtamo liniju te sto su 
# ----tocke blize liniji model je preciznijie
y_test_p = linearModel.predict(X_test_n)
plt.grid()
plt.scatter(y_test, y_test_p)
min_val = min(y_test.min(), y_test_p.min())
max_val = max(y_test.max(), y_test_p.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Idealno (y=x)')
plt.show()

#-----Vrednovanje modela tako da izvrsimo racunanje regresijskih metrica
print(f"Srednja kvadratna pogreska (MSE) {metrics.mean_squared_error(y_test,y_test_p):.2f}")
print(f"Korijen iz srednja kvadratna pogreska (RMSE) {metrics.root_mean_squared_error(y_test,y_test_p):.2f}")
print(f"Srednja absolutna pogreska (MAE) {metrics.mean_absolute_error(y_test,y_test_p):.2f}")
print(f"Srednja absolutna postotna pogreska (MAPE) {metrics.mean_absolute_percentage_error(y_test,y_test_p):.2f}")
print(f"Koeficijen determinacije {metrics.r2_score(y_test,y_test_p):.2f}")
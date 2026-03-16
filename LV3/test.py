import pandas as pd
import matplotlib.pyplot as plt

# 1. Učitavanje podataka
data = pd.read_csv('data_C02_emission.csv')

# 2. Histogram - prikazuje distribuciju potrošnje u gradu
plt.figure(figsize=(8, 5)) # Dodao sam opciju za veličinu prozora
data['Fuel Consumption City (L/100km)'].plot(kind='hist', bins=100, color='skyblue', edgecolor='black')
plt.title('Histogram potrošnje goriva u gradu')
plt.xlabel('L/100 km')
plt.ylabel('Frekvencija')

# 3. Box plot - prikazuje medijan, kvartile i outlier-e (ekstremne vrijednosti)
plt.figure(figsize=(6, 6))
data['Fuel Consumption City (L/100km)'].plot(kind='box')
plt.title('Box plot potrošnje goriva u gradu')
plt.ylabel('L/100 km')

# 4. Prikazivanje oba grafa
plt.show()
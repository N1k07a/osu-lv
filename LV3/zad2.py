import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('data_C02_emission.csv')

# A zadatak
plt.figure()
plt.hist(data['CO2 Emissions (g/km)'], bins=10 , color="blue")
plt.title("Histogram emisije CO2")
plt.ylabel("CO2 Emissions (g/kg)")
plt.ylabel("Broj vozila")
plt.show()

# B zadatak
plt.figure(figsize=(8, 5))
plt.scatter(data[data['Fuel Type'] == 'Z']['Fuel Consumption Comb (L/100km)'], data[data['Fuel Type'] == 'Z']['CO2 Emissions (g/km)'])
plt.scatter(data[data['Fuel Type'] == 'X']['Fuel Consumption Comb (L/100km)'], data[data['Fuel Type'] == 'X']['CO2 Emissions (g/km)'])
plt.scatter(data[data['Fuel Type'] == 'E']['Fuel Consumption Comb (L/100km)'], data[data['Fuel Type'] == 'E']['CO2 Emissions (g/km)'])
plt.scatter(data[data['Fuel Type'] == 'D']['Fuel Consumption Comb (L/100km)'], data[data['Fuel Type'] == 'D']['CO2 Emissions (g/km)'])
plt.scatter(data[data['Fuel Type'] == 'N']['Fuel Consumption Comb (L/100km)'], data[data['Fuel Type'] == 'N']['CO2 Emissions (g/km)'])
plt.show()

# C zadatak
grouped_fuel = data.groupby('Fuel Type')
plt.figure()
data.boxplot(column='Fuel Consumption Hwy (L/100km)', by='Fuel Type')
plt.show()

# D zadatak
plt.figure()
grouped_fuel = grouped_fuel[['Fuel Type']].size()
grouped_fuel.plot(kind="bar")
plt.show()

# E zadatak
plt.figure()
co2_emission = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
co2_emission.plot(kind="bar")
plt.show()
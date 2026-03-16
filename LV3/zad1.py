import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

# A zadatak
print("Zadatak A")
print(f"Broj mjerenja je {len(data)}")
print(f"Tipovi svake velicine je {data.dtypes}")
print(f"Nedstajace vrijednosti {data.isnull().sum()}")
print(f"Duplicirane vrijednosti {data.duplicated().sum()}")

data = data.dropna().drop_duplicates().reset_index(drop = True)

for col in ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']:
    data[col] = data[col].astype('category')

# B zadatak
print("\nZadatak B")
sorted_data = data.sort_values('Fuel Consumption City (L/100km)')
print(f"Minimalna grad. potrosnja \n {sorted_data[['Make', 'Model', 'Fuel Consumption City (L/100km)']].head(3)}")
print(f"Maksimalna grad. potrosnja \n{sorted_data[['Make', 'Model', 'Fuel Consumption City (L/100km)']].tail(3)}")

# C zadatak
print("\nZadatak C")
engine_size = data[ (data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]
print(f"Vozila izmedu 2.5L  i 3.5L ima {len(engine_size)}")
print(f"Mean of CO2 consumation is {engine_size['CO2 Emissions (g/km)'].mean():.2f}")

# D zadatak 
print("\nZadatak D")
audi_car = data[data['Make'] == "Audi"]
print(f"Vozila tipa Audi je je {len(audi_car)}")
print(f"Prosjecna CO2 Audi sa 4 cilindra je {audi_car[(audi_car['Cylinders'] == 4)]['CO2 Emissions (g/km)'].mean():.2f}")

# E zadatak
print("\nZadatak E")
car_with_4_6_8 = data[(data['Cylinders'] == 4) | (data['Cylinders'] == 6) | (data['Cylinders'] == 8)]
print(f"Broj vozila sa 4,6 ili 8 cilindara je {len(car_with_4_6_8)}")
print(f"Prosjecna emisija CO2 za 4 cilindra je {car_with_4_6_8[(car_with_4_6_8['Cylinders'] == 4)]['CO2 Emissions (g/km)'].mean():.2f}")
print(f"Prosjecna emisija CO2 za 6 cilindra je {car_with_4_6_8[(car_with_4_6_8['Cylinders'] == 6)]['CO2 Emissions (g/km)'].mean():.2f}")
print(f"Prosjecna emisija CO2 za 8 cilindra je {car_with_4_6_8[(car_with_4_6_8['Cylinders'] == 8)]['CO2 Emissions (g/km)'].mean():.2f}")

# F zadatak
print("\nZadatak F")
disel = data[data['Fuel Type'] == "D"]
gasoline = data[data['Fuel Type'] == "X"]
print(f"Potrosnja gradska za dizel je {disel['Fuel Consumption City (L/100km)'].mean():.2f}")
print(f"Potrosnja gradska za dizel je {gasoline['Fuel Consumption City (L/100km)'].mean():.2f}")

# G zadatak
print("\nZadatak G")
print(f"Vozilo s 4 cilindra koje ima najvecu gradsku potrosnju je {disel[(disel['Cylinders'] == 4)]['Fuel Consumption City (L/100km)'].max()}")

# H zadatak
print("\nZadatak H")
manual = data[data['Transmission'].str.startswith('M')]
print(f"Vozila sa rucnim mijenjacom je {len(manual)}")

# I zadatak
print("\nZadatak I")
print(f"Koleracija  numerickih velicina je {data.corr(numeric_only = True)}")
from sklearn import datasets
from sklearn . model_selection import train_test_split #Potrebno za razdvajanje test i train skupa
import pandas as pd #Ucitavanje csv datoteke
import matplotlib.pyplot as plt #Crtanje
from sklearn . preprocessing import StandardScaler #Scaliranje standradizacijom
from sklearn . preprocessing import MinMaxScaler #Scaliranje min max
import sklearn . linear_model as lm #Potreban za linearno regresijski model
from sklearn import metrics #Sluzi kako bi mogli nmapraviti evaluaciju modela na temelju metrica
import numpy as np
from sklearn . preprocessing import OneHotEncoder #Sluzi za 1-od-K kodiranje kategorickih velicina
from sklearn.compose import ColumnTransformer
from sklearn.metrics import max_error

dt = pd.read_csv('data_C02_emission.csv')

X = dt[['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)', 'Fuel Type']]
y = dt['CO2 Emissions (g/km)']


# ColumnTransformer koristim kako bi napravio OneHotEncoding na Fuel Type posto su pomjesani numericke i kategoricke vrijednosti, passthrough nam govroi da sve numericke vrijednosti preskoci
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), ['Fuel Type'])
    ],
    remainder='passthrough'
)

X_encode = ct.fit_transform(X)
name_of_columns = ct.get_feature_names_out()

X_encode_df = pd.DataFrame(X_encode, columns=name_of_columns)

X_train, X_test, y_train, y_test = train_test_split(X_encode_df, y, test_size=0.2, random_state=5)

model = lm.LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.grid()
plt.scatter(y_test, y_pred)
plt.show()

#-----Pronalazenje maksimalne pogreske u nekom modelu----
pogreske = np.abs(y_test - y_pred) #Oduzimamo test od predikcije i dobivamo tablicu u kojoj trazimo max error, mogli smo sa max_error fun, ali nisam naso kako da nadem index

max_err = pogreske.max()
id_max_err = pogreske.idxmax()

print(f"Maksimalna pogreske iznosi {max_err:.2f} te se nalazi na {id_max_err} poziciji")
print(f"Radi se o modelu {dt[id_max_err:id_max_err+1]['Model'].values[0]}") #values koristimo kako bi nam vratio samo model, bez toga bi nam vratio i index na koje se nalazi
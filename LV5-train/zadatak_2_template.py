import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay


labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

y_train = y_train.ravel()
y_test = y_test.ravel()

# --- a) Distribucija klasa ---
# Dohvacanje svih vrsta i koliko ih ima
u_train, counts_train = np.unique(y_train, return_counts=True)
u_test, counts_test = np.unique(y_test, return_counts=True)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar([labels[k] for k in u_train], counts_train)
plt.title("Distribucija - Trening")

plt.subplot(1, 2, 2)
plt.bar([labels[k] for k in u_test], counts_test)
plt.title("Distribucija - Test")
plt.show()

# --- b) Izgradnja modela ---
# max_iter povećavamo jer logistička regresija na realnim podacima zahtijeva više koraka za konvergenciju
logReg_model = LogisticRegression()
logReg_model.fit(X_train, y_train)

# --- c) Parametri modela ---
print(f"\nPresjek (Intercept/Theta0): {logReg_model.intercept_}")
print(f"Koeficijenti (Weights/Thetas): \n{logReg_model.coef_}")

# --- d) Regije odluke ---
# Pozivamo funkciju koja je već definirana u tvom kodu
plot_decision_regions(X_train, y_train, classifier=logReg_model)
plt.xlabel('Duljina kljuna (mm)')
plt.ylabel('Duljina peraje (mm)')
plt.legend(loc='upper left')
plt.title("Granice odluke na trening podacima")
plt.show()

# --- e) Evaluacija na testnom skupu ---
y_pred = logReg_model.predict(X_test)

print("Matrica zabune")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

print(f"Preciznost: {accuracy_score(y_test, y_pred)}")
print("Klasifikacijski report")
print(classification_report(y_test, y_pred, target_names=[labels[k] for k in logReg_model.classes_]))

# --- f) Dodavanje ulaznih veličina ---
# Primjer kako dodati 'body_mass_g'
input_variables_prosireno = ['bill_length_mm', 'flipper_length_mm', 'body_mass_g']
X_prosireno = df[input_variables_prosireno].to_numpy()
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_prosireno, y, test_size=0.2, random_state=123)

model_p = LogisticRegression(max_iter=1000)
model_p.fit(X_train_p, y_train_p.ravel())
y_pred_p = model_p.predict(X_test_p)
print(f"\nTočnost s dodatnom značajkom (masa tijela): {accuracy_score(y_test_p, y_pred_p):.2f}")
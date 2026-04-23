import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# a) Prikaz podataka za učenje i testiranje
plt.figure()
plt.scatter(X_train[:,0], X_train[:,1],c=y_train, cmap='RdBu_r', marker='X')
plt.scatter(X_test[:,0], X_test[:,1],c=y_test, cmap='viridis', marker='X')
plt.show()

# b) Izgradnja modela logističke regresije 
model = LogisticRegression()
model.fit(X_train,y_train)

# c) Parametri modela i granica odluke 
theta_0 = model.intercept_[0]
theta_1, theta_2 = model.coef_[0]
print(f"Parametri: theta0={theta_0:.2f}, theta1={theta_1:.2f}, theta2={theta_2:.2f}")

x1_line = np.array([np.min(X_train[:, 0]), np.max(X_train[:, 0])])
x2_line = -(theta_1 * x1_line + theta_0) / theta_2
plt.figure()
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='RdYlBu')
plt.plot(x1_line, x2_line, color='black', linestyle='--', label='Granica odluke')
plt.show()

# d) Klasifikacija i metrike
y_pred = model.predict(X_test)
print("\nMetrike na testnom skupu:")
print(f"Točnost: {accuracy_score(y_test, y_pred):.2f}")
print(f"Preciznost: {precision_score(y_test, y_pred):.2f}")
print(f"Odziv: {recall_score(y_test, y_pred):.2f}")

cm = confusion_matrix(y_test, y_pred) # matrica zabune
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

# e) Prikaz točnih (zeleno) i netočnih (crno) klasifikacija
ispravno = (y_test == y_pred)
plt.scatter(X_test[ispravno,0], X_test[ispravno,1], c='green', label= 'Točnost')
plt.scatter(X_test[~ispravno, 0], X_test[~ispravno, 1], c='black',label = 'Pogrešno')
plt.legend()
plt.show()
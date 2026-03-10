import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)

gender = data[: , 0]
height = data[: , 1]
weight = data[: , 2]

leng = len(data)

print(f"Mjerenja su izvrsena na {leng}")

plt.figure()
plt.scatter(height, weight)
plt.xlabel("Visina")
plt.ylabel("Tezina")
plt.title("Odnos visine i težine")
plt.show()

data_50 = data[::40,:]
plt.figure()
plt.scatter(data_50[:,1], data_50[:,2])
plt.xlabel("Visina svake 50 osobe")
plt.ylabel("Tezina svake 50 osobe")
plt.title("Odnos visine i težine svake 50-te osobe")
plt.show()

print(f"Maksimalna visina je {max(height)} => Minimalna visina je {min(height)} => ::: Srednja vrijednost visine je {height.mean()}")

ind_m = (data[:, 0] == 1) 
ind_z = (data[:, 0] == 0) 

visine_m = data[ind_m, 1]
visine_z = data[ind_z, 1]

print(f"Muški :=> Maksimalna visina je {max(visine_m)} => Minimalna visina je {min(visine_m)} => ::: Srednja vrijednost visine je {visine_m.mean()}")
print(f"Žene :=> Maksimalna visina je {max(visine_z)} => Minimalna visina je {min(visine_z)} => ::: Srednja vrijednost visine je {visine_z.mean()}")
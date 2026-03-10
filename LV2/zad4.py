import numpy as np
import matplotlib.pyplot as plt

black = np.zeros((50,50))
white = np.ones((50,50))

upper_row = np.hstack((black,white))
lower_row = np.hstack((white,black))

whole_img = np.vstack((upper_row,lower_row))

plt.figure()
plt.imshow(whole_img, cmap="gray")
plt.axis([0,100,0,100])
plt.show()
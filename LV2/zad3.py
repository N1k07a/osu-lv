import numpy as np
import matplotlib . pyplot as plt

img = plt . imread ( "road.jpg" )
img = img[:,:,0] . copy()

plt.figure()
plt.imshow(img, cmap="gray", alpha=0.7)
plt.title("Posvijetljen")
plt.show()

plt.figure()
width = img.shape[1]
start = width // 4
end = width // 2
img_quarter = img[:, end:start:-1] 
plt.imshow(img_quarter, cmap="gray")
plt.title("1/4 - 1/2")
plt.show()

plt.figure()
img_rotated = np.rot90(img, k=-1)
plt.imshow(img_rotated, cmap="gray")
plt.title("Rotiran")
plt.show()

plt.figure()
img_mirrored = img[:,::-1]
plt.imshow(img_mirrored, cmap="gray")
plt.title("Zrcalit")
plt.show()
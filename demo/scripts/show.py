#显示不同通道
from matplotlib import pyplot as plt
import cv2

img = cv2.imread('J:/a1_big.png',1)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)
y = img[:,:,0]
print(y.shape)

plt.imshow(y)
plt.show()


plt.imshow(img[:,:,1])
plt.show()
plt.imshow(img[:,:,2])
plt.show()


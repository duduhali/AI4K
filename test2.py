import cv2
import matplotlib.pyplot as plt


img = cv2.imread('E:/ai_data/flower/small_images/image_00001.jpg')
print(img)

plt.imshow(img[..., -1::-1])
plt.show()


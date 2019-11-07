import numpy as np
from keras.utils import np_utils

random_data = np.random.randint(0,10,16)
random_data = np_utils.to_categorical(random_data, 10)  # 独热码
print(random_data.shape)
print(random_data)

random_data2 = np.random.randint(0,10,16)
random_data2 = np_utils.to_categorical(random_data2, 10)  # 独热码
print(random_data2.shape)

input_batch = np.concatenate((random_data, random_data2))
print('shape',input_batch.shape)
print('input_batch',input_batch)




'''输入540p --> 反卷积 --> 4K
    4K --> 图像识别 --> 判断是真4K还是生成的4K
'''

import glob
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,Activation,MaxPool2D,Flatten,Dense,BatchNormalization,Reshape,UpSampling2D
from keras import optimizers
from keras.callbacks import Callback
import matplotlib.pyplot as plt

# 定义生成器模型
def generator_model():
    model = Sequential()
    # model.add(Dense(input_dim=16*16*3, units=128 * 16 * 16))
    # # # model.add(BatchNormalization())  # 批标准化
    # # model.add(Activation("tanh"))
    # model.add(Reshape((16, 16, 128), input_shape=(128 * 16 * 16,)))  # 16, 16 像素

    model.add( UpSampling2D(size=(2, 2), input_shape=(16,16,3)) )  # 输出：32 x 32像素
    # model.add( UpSampling2D(size=(2, 2)))  # 输出：32 x 32像素
    model.add(Conv2D(128, (5, 5), padding="same"))
    model.add(Activation("tanh"))

    model.add(UpSampling2D(size=(2, 2)))  # 输出：64 x 64像素
    model.add(Conv2D(3, (5, 5), padding="same"))
    model.add(Activation("tanh"))
    return model

# 定义判别器模型
def discriminator_model():
    model = Sequential()
    model.add(Conv2D(
        64, # 64 个过滤器，输出的深度（depth）是 64
        (5, 5), # 过滤器在二维的大小是（5 * 5）
        padding='same', # same 表示输出的大小不变，因此需要在外围补零2圈
        input_shape=(64, 64, 3) # 输入形状[64, 64, 3]。3 表示 RGB 三原色
    ))
    model.add(Activation("tanh")) # 添加 Tanh 激活层
    model.add(MaxPool2D(pool_size=(2, 2))) # 池化层
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation("tanh"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation("tanh"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten()) # 扁平化
    model.add(Dense(1024)) # 1024 个神经元的全连接层
    model.add(Activation("tanh"))
    model.add(Dense(1)) # 1 个神经元的全连接层
    model.add(Activation("sigmoid")) # 添加 Sigmoid 激活层

    return model

def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False # 初始时 判别器 不可被训练 这一行很很重要
    model.add(discriminator)
    return model

LEARNING_RATE = 0.0002
BETA_1 = 0.5
EPOCHS = 20
BATCH_SIZE = 128

#训练模型
def train():
    small_images_path = "E:/ai_data/flower/small_images/*"
    big_images_path = "E:/ai_data/flower/images/*"
    small_arr = []
    big_arr = []
    for small_image, big_image in zip(glob.glob(small_images_path), glob.glob(big_images_path)):
        # print(big_image,small_image)
        if small_image.split('\\')[-1] == big_image.split('\\')[-1]:  # 为确保万一，判断下文件名是否相等
            small_arr.append(cv2.imread(small_image))
            big_arr.append(cv2.imread(big_image))
        else:
            print('not like', big_image, small_image)

    small_arr = np.array(small_arr)
    big_arr = np.array(big_arr)
    small_arr = (small_arr.astype(np.float32) - 127.5) / 127.5
    big_arr = (big_arr.astype(np.float32) - 127.5) / 127.5

    print(small_arr.shape)  # (3000, 16, 16, 3)
    small_arr2 = small_arr.reshape(-1, 16 * 16 * 3)
    print(small_arr2.shape)  # (3000, 768)
    print(big_arr.shape)  # (3000, 64, 64, 3)
    small_arr_g = small_arr.copy()

    # 构造 生成器 和 判别器
    g = generator_model()
    d = discriminator_model()

    # 构建 生成器 和 判别器 组成的网络模型
    d_on_g = generator_containing_discriminator(g, d)

    # 优化器用 Adam Optimizer
    g_optimizer = optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)
    d_optimizer = optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)

    '''判别器单独训练，生成器通过串联模型(且判别器不可被训练时)训练'''
    # 配置 生成器 和 判别器
    g.compile(loss="binary_crossentropy", optimizer=g_optimizer)# g模型只用来生成图片，不直接训练，训练由串联模型完成
    d_on_g.compile(loss="binary_crossentropy", optimizer=g_optimizer)
    d.trainable = True
    d.compile(loss="binary_crossentropy", optimizer=d_optimizer)

    results = []
    #判别器，1 为真图  0 为假
    # 开始训练
    for epoch in range(EPOCHS):
        # 打乱数据
        the_index = range(big_arr.shape[0])
        np.random.shuffle(list(the_index))
        big_arr = big_arr[the_index]
        small_arr = small_arr[the_index]

        np.random.shuffle(list(the_index))
        small_arr_g = small_arr_g[the_index]
        for index in range(int(big_arr.shape[0] / BATCH_SIZE)):
            input_batch = big_arr[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]
            original_data = small_arr[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]
            generated_images = g.predict(original_data, verbose=0) #生成图片数据

            input_batch = np.concatenate((input_batch, generated_images))
            output_batch = [1] * BATCH_SIZE + [0] * BATCH_SIZE #一维数组
            # 训练判别器，让它具备识别不合格生成图片的能力
            d_loss = d.train_on_batch(input_batch, output_batch)

            '''训练生成器：
                让判别器不可被训练
                让随机数 通过d_on_g后被当成真图
            '''
            d.trainable = False
            # 训练生成器，并通过不可被训练的判别器去判别
            original_data_g = small_arr_g[index * BATCH_SIZE: (index + 1) * BATCH_SIZE]
            g_loss = d_on_g.train_on_batch(original_data_g, [1] * BATCH_SIZE)
            d.trainable = True # 恢复判别器可被训练

            # 打印损失
            print("Step %d Generator Loss: %f Discriminator Loss: %f" % (index, g_loss, d_loss))
            results.append([g_loss, d_loss])

        print(epoch)
        if epoch % 10 == 9:
            g.save_weights("./generator_weight.h5", True)
            d.save_weights("./discriminator_weight.h5", True)

    # 损失曲线
    plt.figure(figsize=(10, 7))
    plt.plot([i[0] for i in results], '.', label='Generator', alpha=0.5)
    plt.plot([i[1] for i in results], '.', label='Discreminator', alpha=0.5)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# 生成图片
def generate():
    g = generator_model() # 构造生成器
    g.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)) # 配置生成器
    g.load_weights("./generator_weight.h5") # 加载训练好的生成器参数

    len = 2
    data = []
    for i in range(len * len):
        data.append(cv2.imread('E:/ai_data/flower/small_images/image_00001.jpg'))
    data = np.array(data)
    # data = data.reshape(-1, 16 * 16 * 3)

    images = g.predict(data, verbose=1)
    images = images * 127.5 + 127.5
    images = images.astype(np.uint8)
    print(images.shape)
    print(images[0])
    print(images[0][..., -1::-1])

    plt.figure(figsize=(70, 40))
    for i in range(len):
        plt.subplot(1, 2, i + 1)
        plt.axis('off')
        plt.imshow(images[i][..., -1::-1])
    plt.show()

if __name__ == "__main__":
    # train()
    generate()

from keras.models import Sequential
from keras.layers import Conv2D,Activation,MaxPool2D,Flatten,Dense,BatchNormalization,Reshape,UpSampling2D
from keras.optimizers import Adam

import glob
from scipy import misc
import numpy as np
from PIL import Image

from keras.datasets import mnist

# 定义判别器模型
def discriminator_model():
    model = Sequential()
    model.add(Conv2D(
        28, # 过滤器，输出的深度（depth）
        (3, 3), # 过滤器在二维的大小是
        padding='same', # same 表示输出的大小不变，因此需要在外围补零2圈
        input_shape=(28, 28, 1) # 输入形状
    ))
    model.add(Activation("tanh")) # 添加 Tanh 激活层
    model.add(MaxPool2D(pool_size=(2, 2))) # 池化层

    model.add(Conv2D(56, (3, 3)))
    model.add(Activation("tanh"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten()) # 扁平化
    model.add(Dense(512)) # 全连接层
    model.add(Activation("tanh"))

    model.add(Dense(1)) # 1 个神经元的全连接层
    model.add(Activation("sigmoid")) # 添加 Sigmoid 激活层

    return model

# 定义生成器模型
def generator_model(): # 从随机数来生成图片
    model = Sequential()
    # 输入的维度是 100, 输出维度（神经元个数）是1024 的全连接层
    model.add(Dense(input_dim=10, units=512))
    model.add(Activation("tanh"))

    model.add(Dense(56 * 7 * 7)) # 全连接层
    model.add(BatchNormalization()) # 批标准化
    model.add(Activation("tanh"))

    model.add(Reshape((7, 7, 56))) # 7 x 7 像素
    model.add(UpSampling2D(size=(2, 2))) # 14 x 14像素 池化的反操作
    model.add(Conv2D(56, (3, 3), padding="same"))
    model.add(Activation("tanh"))

    model.add(UpSampling2D(size=(2, 2))) # 28 x 28像素
    model.add(Conv2D(1, (3, 3), padding="same"))
    model.add(Activation("tanh"))

    return model


# 构造包含生成器和判别器的串联模型
# 输入 -> 生成器 -> 判别器 -> 输出
def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False # 初始时 判别器 不可被训练 ######################这一行貌似可去掉
    model.add(discriminator)
    return model


# Hyperparameters 超参数
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
BETA_1 = 0.5

#训练模型
def train():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(X_train[0].shape)  # (28, 28)
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

    print(X_train[0].shape)  # (28, 28, 1)
    print(X_train.shape)  # (60000, 28, 28, 1)
    print(y_train.shape)  # (60000,)
    print(y_test.shape)  # (10000,)

    # X_train [0, 255]
    # X_train /= 255 #relu  [0,x]
    X_train = (X_train - 127.5) / 127.5  # tanh [-1,1]


    # 构造 生成器 和 判别器
    g = generator_model()
    d = discriminator_model()

    # 构建 生成器 和 判别器 组成的网络模型
    d_on_g = generator_containing_discriminator(g, d)

    # 优化器用 Adam Optimizer
    g_optimizer = Adam(lr=LEARNING_RATE, beta_1=BETA_1)
    d_optimizer = Adam(lr=LEARNING_RATE, beta_1=BETA_1)

    '''判别器单独训练，生成器通过串联模型(且判别器不可被训练时)训练'''
    # 配置 生成器 和 判别器
    g.compile(loss="binary_crossentropy", optimizer=g_optimizer)# g模型只用来生成图片，不直接训练，训练由串联模型完成
    d_on_g.compile(loss="binary_crossentropy", optimizer=g_optimizer)
    d.trainable = True
    d.compile(loss="binary_crossentropy", optimizer=d_optimizer)

    #判别器，1 为真图  0 为假
    # 开始训练
    for epoch in range(EPOCHS):
        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            input_batch = X_train[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]

            # 连续型均匀分布的随机数据（噪声）
            random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 10))
            generated_images = g.predict(random_data, verbose=0) #生成图片数据

            input_batch = np.concatenate((input_batch, generated_images))
            output_batch = [1] * BATCH_SIZE + [0] * BATCH_SIZE #一维数组
            # 训练判别器，让它具备识别不合格生成图片的能力
            d_loss = d.train_on_batch(input_batch, output_batch)

            '''训练生成器：
                让判别器不可被训练
                让随机数 通过d_on_g后被当成真图
            '''
            d.trainable = False
            random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 10))
            # 训练生成器，并通过不可被训练的判别器去判别
            g_loss = d_on_g.train_on_batch(random_data, [1] * BATCH_SIZE)
            d.trainable = True # 恢复判别器可被训练

            # 打印损失
            print("Step %d Generator Loss: %f Discriminator Loss: %f" % (index, g_loss, d_loss))

        print(epoch)
        # 保存 生成器 和 判别器 的参数
        if epoch % 3 == 2:
        # if epoch % 10 == 9:
            # g.save_weights("generator_weight2", True) #用tf.keras 在mac上这样保存就行，在windows上要指定HDF5格式
            # d.save_weights("discriminator_weight2", True)
            # 以HDF5格式保存权重，不然使用权重时会比较麻烦
            g.save_weights("./generator_weight.h5", True)
            d.save_weights("./discriminator_weight.h5", True)

if __name__ == "__main__":
    train()
    # generate()
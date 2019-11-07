# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D,Activation,MaxPool2D,Flatten,Dense,BatchNormalization,Reshape,UpSampling2D
# from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D,Activation,MaxPool2D,Flatten,Dense,BatchNormalization,Reshape,UpSampling2D
from keras.optimizers import Adam


import glob
from scipy import misc
import numpy as np
from PIL import Image


# Hyperparameters 超参数
EPOCHS = 10
BATCH_SIZE = 512
LEARNING_RATE = 0.0002
BETA_1 = 0.5

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

# 定义生成器模型
def generator_model(): # 从随机数来生成图片
    model = Sequential()
    # 输入的维度是 100, 输出维度（神经元个数）是1024 的全连接层
    model.add(Dense(input_dim=100, units=1024))
    model.add(Activation("tanh"))
    model.add(Dense(128 * 8 * 8)) # 8192 个神经元的全连接层
    model.add(BatchNormalization()) # 批标准化
    model.add(Activation("tanh"))
    model.add(Reshape((8, 8, 128), input_shape=(128 * 8 * 8, ))) # 8 x 8 像素
    model.add(UpSampling2D(size=(2, 2))) # 16 x 16像素 池化的反操作
    model.add(Conv2D(128, (5, 5), padding="same"))
    model.add(Activation("tanh"))
    model.add(UpSampling2D(size=(2, 2))) # 32 x 32像素
    model.add(Conv2D(128, (5, 5), padding="same"))
    model.add(Activation("tanh"))
    model.add(UpSampling2D(size=(2, 2))) # 64 x 64像素
    model.add(Conv2D(3, (5, 5), padding="same"))
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

#训练模型
def train():
    data = []
    # 获取训练数据  glob查找符合特定规则的文件路径名
    for image in glob.glob("E:/data/images/*"):
        image_data = misc.imread(image) # imread 利用 PIL 来读取图片数据
        data.append(image_data)
    input_data = np.array(data)
    # input_data.shape = (3000, 64, 64, 3)
    # 将数据标准化成 [-1, 1] 的取值, 这也是 Tanh 激活函数的输出范围
    input_data = (input_data.astype(np.float32) - 127.5) / 127.5

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
        for index in range(int(input_data.shape[0] / BATCH_SIZE)):
            input_batch = input_data[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]

            # 连续型均匀分布的随机数据（噪声）
            random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
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
            random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
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

# 生成图片
def generate():
    g = generator_model() # 构造生成器
    g.compile(loss="binary_crossentropy", optimizer=Adam(lr=LEARNING_RATE, beta_1=BETA_1)) # 配置生成器
    g.load_weights("./generator_weight.h5") # 加载训练好的生成器参数

    # 连续型均匀分布的随机数据（噪声） 生成-1到1到随机数，纬度是(BATCH_SIZE, 100)
    random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 10))
    # 用随机数据作为输入，生成图片数据
    images = g.predict(random_data, verbose=1)
    # 用生成的图片数据生成 PNG 图片
    for i in range(BATCH_SIZE):
        # print(images[i]) # 范围在-1到1之间的矩阵
        image = images[i] * 127.5 + 127.5
        # print(image) # 范围在0到255之间的矩阵
        Image.fromarray(image.astype(np.uint8)).save("image-%s.png" % i) #保存图片
        # break

def imshow(inp, title=None, ax=None):
    # 在屏幕上绘制图像
    """Imshow for Tensor."""
    if inp.size()[0] > 1:
        inp = inp.numpy().transpose((1, 2, 0))
    else:
        inp = inp[0].numpy()
    mvalue = np.amin(inp)
    maxvalue = np.amax(inp)
    if maxvalue > mvalue:
        inp = (inp - mvalue)/(maxvalue - mvalue)
    ax.imshow(inp)
    if title is not None:
        ax.set_title(title)

if __name__ == "__main__":
    train()
    # generate()

    # import matplotlib.pyplot as plt
    # batch_size = 64
    # fig = plt.figure(figsize=(15, 15))
    # f, axarr = plt.subplots(8, 8, sharex=True, figsize=(15, 15))
    # for i in range(batch_size):
    #     axarr[i // 8, i % 8].axis('off')
    #     imshow(img[i], samples.data.numpy()[i][0, 0, 0].astype(int), axarr[i // 8, i % 8])
    # plt.show()



    # plt.figure(figsize=(10, 7))
    # # 有四个特征图，循环把它们打印出来
    # for i in range(4):
    #     plt.subplot(1, 4, i + 1)
    #     plt.axis('off')
    #     # feature_maps[0].data.numpy().shape =  (1, 4, 28, 28)
    #     plt.imshow(feature_maps[0][0, i, ...].data.numpy())
    # plt.show()
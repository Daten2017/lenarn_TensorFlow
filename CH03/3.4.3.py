import tensorflow as tf

# 声明w1、w2两个变量。这里还通过seed参数设定了随机种子，
# 这样可以保证每次运行得到的结果是一样的。
w1 = tf.Variable(tf.random.normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random.normal([3, 1], stddev=1, seed=1))

# 暂时将输入的特征向量定义为一个常量。注意这里x要是1x2的矩阵
x = tf.constant([0.7, 0.9])
x = tf.reshape(x, (1, 2))

# 通过3.4.2节描述的前向传播算法获得神经网络的输出。
a = x @ w1
y = a @ w2

# 输出y的结果
tf.print(y)

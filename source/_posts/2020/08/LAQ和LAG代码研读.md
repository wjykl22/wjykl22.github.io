---
title: LAQ和LAG代码研读
mathjax: true
abbrlink: d34fba27
date: 2020-08-06 15:53:47
password: wjykl22
abstract: 此文加密，请输入密码
message: 请输入密码
tags:
	- 分布式机器学习
	- 通信优化
	- 梯度压缩
	- 科研
categories:
	- 科研
	- 分布式机器学习
	- 通信优化
	- 梯度压缩
---

## LAQ实验源码

LAQ实验使用Tensorflow框架编写，对MNIST手写数据集进行梯度下降算法实验，本文将对该实验代码做详细解读，辅助理解LAQ算法思想和编程思想。

### 普通变量

| 变量  | 论文中的含义                          |
| ----- | ------------------------------------- |
| Iter  | 迭代次数$K$                           |
| C     | 限制某个工作节点不能够超过$C$次不上传 |
| ck    | 设置为0.8，权重的衰减系数             |
| Loss  | 记录下每次迭代的误差值                |
| b     | 量化bit范围                           |
| alpha | 步长$ \alpha$                         |
| cl    | 正则化因子                            |
| M     | 工作节点数量，论文中$M=10$            |
| D     | 考虑过去的10次迭代梯度。$D=10$        |
| le    | 神经网络层数                          |
| nv    | 模型当中的参数个数                    |

### 重要变量含义

| 变量   | 论文中的含义                                                 |
| ------ | ------------------------------------------------------------ |
| e      | $\varepsilon_{m}^{k}$，当前轮次的量化误差，形状：(M, )       |
| ehat   | $\hat{\varepsilon}_{m}^{k-1}$，上一轮次的量化误差，形状：(M, ) |
| dtheta | $\boldsymbol{\theta}^{k+1-d}-\boldsymbol{\theta}^{k-d}$，形状：(nv, Iter) |
| theta  | $\boldsymbol{\theta}^{k}$，记录下当前的轮次的梯度            |
| dL     | $\delta Q_{m}^{k}=Q_{m}\left(\boldsymbol{\theta}^{k}\right)-Q_{m}\left(\hat{\boldsymbol{\theta}}_{m}^{k-1}\right)$，其中gr为$Q_{m}\left(\boldsymbol{\theta}^{k}\right)$，而mgr为$Q_{m}\left(\hat{\boldsymbol{\theta}}_{m}^{k-1}\right)$，形状(M, nv) |
| dsa    | $\left.\sum_{m \in \mathcal{M}_{c}^{k}}\left(Q_{m}\left(\hat{\boldsymbol{\theta}}^{k-1}\right)-Q_{m}\left(\boldsymbol{\theta}^{k}\right)\right)\right]$，形状：(nv, ) |
| ksi    | 表示为$\xi_{1} \geq \xi_{2} \geq \cdots \geq \xi_{D}$        |

### 伪代码



### 代码详解

#### 方法

##### 梯度或参数向量化：gradtovec(grad)

```python
def gradtovec(grad):
    vec = np.array([])
    # 记录该神经网络有多少层，遍历每一层，将权重取出来
    le = len(grad)
    for i in range(0, le):
        # 取到该层侵权中
        a = grad[i]
        # 从tf张量转化为numpy数组
        b = a.numpy()
        # 展开成向量
        if len(a.shape) == 2:
            da = int(a.shape[0])
            db = int(a.shape[1])
            b = b.reshape(da * db)
          else:
             da = int(a.shape[0])
             b.reshape(da)
        # 将每一层权重顺次拼接成完整权重
        vec = np.concatenate((vec, b), axis=0)
    return vec
```

> `np.concatenate`：用于拼接数组

##### 向量梯度或权重化

该方法是上述方法的反向过程，压缩为Tensorflow的特定格式，以便进行梯度下降更新权重。

```python
def vectograd(vec, grad):
    # grad是tf格式的梯度，最终要将vec压缩至grad形状
    le = len(grad)
    for i in range(0, le):
        a = grad[i]
        b = a.numpy()
        if len(a.shape) == 2:
            da = int(a.shape[0])
            db = int(a.shape[1])
            c = vec[0:da * db]
            c = c.reshape(da, db)
            lev = len(vec)
            vec = vec[da * db:lev]
        else:
            da = int(a.shape[0])
            c = vec[0:da]
            lev = len(vec)
            vec = vec[da:lev]
        grad[i] = 0 * grad[i] + c
    return grad
```

##### 梯度向量量化

r: $R_{m}^{k}=\left\|\nabla f_{m}\left(\boldsymbol{\theta}^{k}\right)-Q_{m}\left(\hat{\boldsymbol{\theta}}_{m}^{k-1}\right)\right\|_{\infty}$

delta: $2 \tau R_{m}^{k}$其中$\tau:=1 /\left(2^{b}-1\right)$

quantv:$Q_{m}\left(\boldsymbol{\theta}^{k}\right)$ 一共包括了如下三个式子
$$
\left.\left[q_{m}\left(\boldsymbol{\theta}^{k}\right)\right]_{i}=\mid \frac{\left[\nabla f_{m}\left(\boldsymbol{\theta}^{k}\right)\right]_{i}-\left[Q_{m}\left(\hat{\boldsymbol{\theta}}_{m}^{k-1}\right)\right]_{i}+R_{m}^{k}}{2 \tau R_{m}^{k}}+\frac{1}{2}\right\rfloor, \quad i=1, \cdots, p
$$

$$
\delta Q_{m}^{k}=Q_{m}\left(\boldsymbol{\theta}^{k}\right)-Q_{m}\left(\hat{\boldsymbol{\theta}}_{m}^{k-1}\right)=2 \tau R_{m}^{k} q_{m}\left(\boldsymbol{\theta}^{k}\right)-R_{m}^{k} \mathbf{1}
$$

$$
Q_{m}\left(\boldsymbol{\theta}^{k}\right)=Q_{m}\left(\hat{\boldsymbol{\theta}}_{m}^{k-1}\right)+\delta Q_{m}^{k}
$$

```python
def quantd(vec, v2, b):
    n = len(vec)
    r = max(abs(vec - v2))
    delta = r / (np.floor(2 ** b) - 1)
    quantv = v2 - r + 2 * delta * np.floor((vec - v2 + r + delta) / (2 * delta))
    return quantv
```

#### 程序过程

##### 初始化数据集

加载

```python
(mnist_images, mnist_labels), (mnist_ta, mnist_tb) = tf.keras.datasets.mnist.load_data()
```

预处理

```python
mnist_ta = mnist_ta / 255
```

将训练数据集分配给到10台工作节点上

```python
for m in range(0, M):
    datr = tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[m * Mi:(m + 1) * Mi, tf.newaxis] / 255, tf.float32),
         tf.cast(mnist_labels[m * Mi:(m + 1) * Mi], tf.int64)))
    datr = datr.batch(Mi)
    Datatr[m] = datr
```

将测试数据集使用DataSet包装

```python
# Datate是训练集数据，这个部分只需要一份就可以了
Datate = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_ta[..., tf.newaxis], tf.float32),
     tf.cast(mnist_tb, tf.int64)))
Datate = Datate.batch(1)

# 将原来的标签数据转换成为One-Hot类型的编码
mnistl = np.eye(10)[mnist_tb]
```

##### 定义网络和优化器

```python
# 这里在定义神经网络
regularizer = tf.contrib.layers.l2_regularizer(scale=0.9)
tf.random.set_random_seed(1234)
mnist_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # tf.keras.layers.Dense(512, activation=tf.nn.relu),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(120, activation=tf.nn.relu,kernel_regularizer=regularizer),
    # tf.keras.layers.Dense(200, activation=tf.nn.relu),
    # tf.keras.layers.Dense(10, kernel_regularizer=regularizer)
    tf.keras.layers.Dense(10)
])

# 定义优化器，梯度下降算法，交叉熵损失函数
mnist_model.compile(optimizer=tf.train.GradientDescentOptimizer(alpha),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])


# optimizer = tf.train.AdamOptimizer()
optimizer = tf.train.GradientDescentOptimizer(alpha)
```

统计模型一共有多少参数

```python
# 在统计模型当中一共有多少各参数
for i in range(0, le):
    a = mnist_model.trainable_variables[i]
    if (len(a.shape) == 2):
        da = int(a.shape[0])
        db = int(a.shape[1])
        nv = nv + da * db
    if (len(a.shape) == 1):
        da = int(a.shape[0])
        nv = nv + da
```

将$\xi_{1} \geq \xi_{2} \geq \cdots \geq \xi_{D}$这些常数初始化完成

```python
for i in range(0, D + 1):
    if (i == 0):
        ksi[:, i] = np.ones(D);
    if (i <= D and i > 0):
        ksi[:, i] = 1 / i * np.ones(D);
        
ksi = ck * ksi
```

##### 训练过程

迭代进行如下循环操作

###### 量化过程

1. 将每一轮的$\boldsymbol{\theta}^{k+1-d}-\boldsymbol{\theta}^{k-d}$都记录下来（本轮和前一轮参数的差值）

```python
    if k == 0:
        thetat = gradtovec(mnist_model.trainable_variables)
    if k >= 1:
        thetat = var
    me = np.zeros(M)
    var = gradtovec(mnist_model.trainable_variables)
    if (k >= 1):
        dtheta[:, k] = var - theta
    theta = var
```

各个工作节点进行如下操作

```python
        for (batch, (images, labels)) in enumerate(Datatr[m].take(1)):
            with tf.GradientTape() as tape:
                logits = mnist_model(images, training=True)
                loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)
                for i in range(0, len(mnist_model.trainable_variables)):
                    if i == 0:
                        l2_loss = cl * tf.nn.l2_loss(mnist_model.trainable_variables[i])
                    if i >= 1:
                        l2_loss = l2_loss + cl * tf.nn.l2_loss(mnist_model.trainable_variables[i])

                # l2_loss = tf.losses.get_regularization_loss()
                loss_value = loss_value + l2_loss
            # 计算梯度
            grads = tape.gradient(loss_value, mnist_model.trainable_variables)
            # 将所得梯度向量化
            vec = gradtovec(grads)
            bvec = vec

            # 对向量梯度进行量化
            vec = quantd(vec, mgr[m, :], b)
            
            # 本轮量化梯度
            gr[m, :] = vec
            
            # 计算量化误差
            dvec = vec - bvec
            e[m] = (dvec.dot(dvec))
```

###### LAG过程

me：$\sum_{d=1}^{D} \xi_{d}\left\|\theta^{k+1-d}-\theta^{k-d}\right\|_{2}^{2}$

下面的步骤执行如下公式：
$$
\left\|Q_{m}\left(\hat{\boldsymbol{\theta}}_{m}^{k-1}\right)-Q_{m}\left(\boldsymbol{\theta}^{k}\right)\right\|_{2}^{2} \leq \frac{1}{\alpha^{2} M^{2}} \sum_{d=1}^{D} \xi_{d}\left\|\boldsymbol{\theta}^{k+1-d}-\boldsymbol{\theta}^{k-d}\right\|_{2}^{2}+3\left(\left\|\boldsymbol{\varepsilon}_{m}^{k}\right\|_{2}^{2}+\left\|\hat{\boldsymbol{\varepsilon}}_{m}^{k-1}\right\|_{2}^{2}\right)
$$

```python
        for d in range(0, D):
            if (k - d >= 0):
                if (k <= D):
                    me[m] = me[m] + ksi[d, k] * dtheta[:, k - d].dot(dtheta[:, k - d])
                if (k > D):
                    me[m] = me[m] + ksi[d, D] * dtheta[:, k - d].dot(dtheta[:, k - d])
        dL[m, :] = gr[m, :] - mgr[m, :]
        if ((dL[m, :].dot(dL[m, :])) >= (1 / (alpha ** 2 * M ** 2)) * me[m] + 3 * (e[m] + ehat[m]) or clock[
            m] == C):
            Ind[m, k] = 1

        # 上传梯度
        if (Ind[m, k] == 1):
            # 前一次量化梯度记为这次的量化梯度
            mgr[m, :] = gr[m, :]
            # 前一次的量化误差更新为这次的量化误差
            ehat[m] = e[m]
            clock[m] = 0
            # 聚合量化梯度变化
            dsa = dsa + dL[m, :]
        if (Ind[m, k] == 0):
            clock[m] = clock[m] + 1

        # 记录每次迭代的损失函数
        if m == 0:
            g = grads
            loss = loss_value.numpy() / M
        else:
            g = [a + b for a, b in zip(g, grads)]
            loss = loss + loss_value.numpy() / M
```

###### 更新权重过程

```python
# 将loss记录下来，为了方便画图
lossfr2[k] = l2_loss
    # lossfg[k]=loss
    lossfr[k] = 0
    for i in range(0, len(mnist_model.trainable_variables)):
        # loss= loss+ cl * np.linalg.norm(mnist_model.trainable_variables[i].numpy()) ** 2
        lossfr[k] = lossfr[k] + cl * np.linalg.norm(mnist_model.trainable_variables[i].numpy()) ** 2
    loss_history[k] = loss

    # 将量化梯度向量重新变成tf的grad形式
    # dsa=dsa-beta/alpha*(theta-thetat)
    ccgrads = vectograd(dsa, grads)
    # grr = copy.deepcopy(mnist_model.trainable_variables)
    # grr2 = [c * cl * 2 for c in grr]
    # ccgrads = [a + b for a, b in zip(cgrads, grr2)]
    # ccgrads=cgrads
    for i in range(0, len(ccgrads)):
        # 计算梯度正则化
        grnorm[k] = grnorm[k] + tf.nn.l2_loss(ccgrads[i]).numpy()
        # 进行一步优化
    optimizer.apply_gradients(zip(ccgrads, mnist_model.trainable_variables),
                              global_step=tf.train.get_or_create_global_step())
```


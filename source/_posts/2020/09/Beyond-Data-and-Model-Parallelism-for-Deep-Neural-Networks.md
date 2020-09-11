---
title: Beyond Data and Model Parallelism for Deep Neural Networks
mathjax: true
date: 2020-09-09 20:29:02
tags:
	- 分布式机器学习
	- 数据并行
	- 模型并行
	- 科研
categories:
	- 科研
	- 分布式机器学习
	- 论文阅读笔记
---

- 题目：深度神经网络的数据和模型并行性

  英文：[Beyond Data and Model Parallelism for Deep Neural Networks](https://arxiv.org/pdf/1807.05358)

- 作者：Zhihao Jia 斯坦福大学

## 摘要

本文定义了一种更宽泛的DNN并行搜索策略，称为SOAP，它将从样本（Sample），操作（Opration）、属性（Attribute）以及参数（Parameter）这四个维度对最优的并行策略进行探索和优化。同时本文还提出了FlexFlow，在SOAP范围内随机搜索并行策略的方法。为了加速搜索，FlexFlow提出了一种新的执行模拟器，比需要对每种并行策略都执行一遍的这类方法速度提高了三个数量级。

## 相关工作

### 数据并行（data parallelism）

对于参数较少，且计算集中DNN操作的有较高的性能，但是对于具有大量参数的操作性能次优（例如：矩阵乘法）。

数据并行[28]广泛应用于现有的深度学习系统（例如：Tensorflow、Caffe2以及Pytorch），但数据并行需要在每个工作节点上都复制一份完整网络，在参数量较大的情况下将成为性能瓶颈。

### 模型并行（model parallelism）

模型并行[15]消除了参数在设备之间同步的开销，但是需要在操作时的数据转换，并且不允许内部操作并行化。

### 专门设计策略（expert-desgned strategies）

根据程序员的经验和知识手动优化并行策略，例如[27] [42]对卷积层和池化层采用数据并行，对全连接层转换为模型并行加速训练卷积神经网络。这样的方法相比于数据和模型并行提高了性能，但是依然是次优的。

### 自动化并行策略

在特定的搜索域内自动化寻找最优的并行策略，例如REINFORCE采用强化学习模型在实际设备上运行不同策略来为模型并行学习得到高效操作策略。但是该方法通过对每种并行策略执行一次迭代来衡量执行时间的方式依然非常低效。再比如OptCNN[25]为DNN的并行化设计线性图，自动找到在每个DNN操作中利用并行性的策略。

现有的自动化框架只是探究了不同操作的并行化（例如REINFORCE[33]）或者在单个操作中的并行（例如OptCNN），忽略了在两个维度上都使用并行性的更快策略。

下表概括了现有方法的并行维度，和这些方法相比，FlexFlow考虑了更多的维度，通常更有效的策略是并行化单个操作。

### 基于图的集群调度

Quincy将任务调度映射到流网络，并使用最小代价最大流(MCMF)算法来找到有效的任务分配；Firmament使用多种MCMF优化算法对Quincy进行了推广，减少了任务放置的延迟。现有的基于图的调度是在任务图是固定的假设之下的，然而，FlexFlow解决了一个不同的问题，即需要联合优化如何利用SOAP维中的并行性将操作划分为任务，以及如何将任务分配给设备。

## 主要贡献

1. 为并行DNN应用定义了SOAP搜索空间，它包括了样本（Sample），操作（Opration）、属性（Attribute）以及参数（Parameter）这四个维度下的并行策略。
2. 在合理假设下，DNN的并行仿真所用时间比DNN直接在真实硬件系统下运行时间少三个数量级
3. 描述了FlexFlow框架，能够从SOAP空间寻找并行执行策略，从而加速DNN训练
4. FlewFlow与现有框架相比，能够提升3.8倍的训练吞吐量，并且提升了可扩展性。

## 架构概述

和现有的学习系统相似，FlexFlow采用操作图$\mathcal{G}$描述所有的DNN操作和状态。每个节点$o_{i} \in \mathcal{G}$都表示一个操作（例如：矩阵乘法和卷积等），每个边$(o_i, o_j) \in \mathcal{G}$表示输出$o_i$和输入$o_j$的张量。FlexFlow将设备拓扑结构记为$\mathcal{D} = \left(\mathcal{D}_{N}, \mathcal{D}_{E}\right)$，描述了所有可能的硬件设备和它们的内在联系，如下图所示。每个节点$d_{i} \in \mathcal{D}_{N}$代表设备（例如CPU或者是GPU），每条边$\left(d_{i}, d_{j}\right) \in \mathcal{D}_{E}$代表硬件之间的连接（例如NVLink, PCI-e或者是网络连接）。每条边都标注有带宽或者延迟。FlexFlow会为操作图和设备拓扑找到一个并行策略，对比现有的框架，FlexFlow具有如下优点：

1. 可编程性：对于在拓扑结构非常深的集群上运行的复杂操作图，设计高效的操作是非常困难的，FlexFlow可以找到高效的并行策略，并提供更加丰富的可编程接口。
2. 可移植性，一种并行策略在一个集群中适用，但在别的集群中是不适用的，而然，FlewFlow可以自动根据集群中硬件的配置，根据应用特点自动寻找适合的并行策略。

<img src="Beyond-Data-and-Model-Parallelism-for-Deep-Neural-Networks/image-20200910084035262.png" alt="image-20200910084035262" style="zoom: 80%;" />

#### FlexFlow架构

其架构主要如上图所示，FlexFlow的执行优化器（Execution Optimizer）将操作图和设备拓扑作为输入，自动生成高效的并行策略。优化器采用MCMC蒙特卡罗搜索方法，找到可能存在的并行策略空间，迭代地找出候选策略并交给执行仿真器（Execution Simulator）。执行仿真采用三角执行算法（delta simulation algorithm）使用对以前模拟的增量更新来模拟新的策略。模拟的执行时间指导搜索生成未来候选人，当搜索时间预算好景，执行优化器将发现的最佳策略发送到分布式运行（Distributed Runtime），来执行实际的并行程序。

#### 局限性

该方式最大的局限性在于假设每个操作的执行时间是可预测的，并且和输入样本是独立的。因此该方法并不适用于执行时间和输入样本相关的应用。然而DNN应用是满足该要求的。并且乘法运算的执行时间是高度独立且可预测的。



> 以上是初步的阅读，未完...
> 

<div id="refer-anchor-1"></div>
- [1] [Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks[C]//Advances in neural information processing systems. 2012: 1097-1105.](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
<div id="refer-anchor-2"></div>
- [2] [Dean J, Corrado G, Monga R, et al. Large scale distributed deep networks[C]//Advances in neural information processing systems. 2012: 1223-1231.](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf)
<div id="refer-anchor-3"></div>
- [3] [Krizhevsky A. One weird trick for parallelizing convolutional neural networks[J]. arXiv preprint arXiv:1404.5997, 2014.](https://arxiv.org/pdf/1404.5997)
<div id="refer-anchor-4"></div>
- [4] [Wu Y, Schuster M, Chen Z, et al. Google's neural machine translation system: Bridging the gap between human and machine translation[J]. arXiv preprint arXiv:1609.08144, 2016.](https://arxiv.org/pdf/1609.08144.pdf%20(7))
<div id="refer-anchor-5"></div>
- [5] [Jia Z, Lin S, Qi C R, et al. Exploring hidden dimensions in parallelizing convolutional neural networks[J]. arXiv preprint arXiv:1802.04924, 2018.](https://arxiv.org/abs/1802.04924)
<div id="refer-anchor-6"></div>
- [6] [Mirhoseini A, Pham H, Le Q V, et al. Device placement optimization with reinforcement learning[J]. arXiv preprint arXiv:1706.04972, 2017.](https://patentimages.storage.googleapis.com/2b/03/41/324a4ae429b203/US10692003.pdf)

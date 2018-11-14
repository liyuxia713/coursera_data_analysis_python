
# Machine Learning

<!-- TOC -->

- [Machine Learning](#machine-learning)
    - [模型评价](#模型评价)
        - [1. 分类模型评价](#1-分类模型评价)
        - [2. 回归模型评价](#2-回归模型评价)
    - [模型介绍](#模型介绍)
        - [1. 机器学习思想](#1-机器学习思想)
            - [`KNN`](#knn)
        - [2. 线性模型](#2-线性模型)
            - [`Least Squared Regeression` `（最小二乘估计）`](#least-squared-regeression-最小二乘估计)
            - [`Ridge Regeression` `岭回归`](#ridge-regeression-岭回归)
            - [`Lasso Regeression`](#lasso-regeression)
            - [`Logistic Regeression`](#logistic-regeression)
            - [`Linear Support Vector Machine ` `线性支持向量机`](#linear-support-vector-machine--线性支持向量机)
            - [线性模型小结](#线性模型小结)
        - [3. 非线性模型](#3-非线性模型)
            - [`Kernalized Support Vector Machine`](#kernalized-support-vector-machine)
            - [`Redial Basis Function Kernel` `径向基核函数`](#redial-basis-function-kernel-径向基核函数)
            - [`Decision Tree` `决策树`](#decision-tree-决策树)
    - [特征工程](#特征工程)

<!-- /TOC -->

## 模型评价

### 1. 分类模型评价
准确率 F-score

### 2. 回归模型评价

`R-Squared Value`

R平方， 也称作决定系数，在线性回归模型中等于相关系数的平方， 一般理解为$1-残差平方和/样本方差$。 
介于[0,1], 越大越好。

## 模型介绍

### 1. 机器学习思想

#### `KNN`
用N临近投票或加权平均的算法，简单计数。

- 简单，计数即可，不用模型。
- 预测效果较好
- 不能很好的抽象关系
- 不稳定（噪声影响大）

### 2. 线性模型

#### `Least Squared Regeression` `（最小二乘估计）`

最小化偏差平方和的方法

- 能很好的抽象关系
- 效果比KNN相对差
- 稳定性比KNN好

#### `Ridge Regeression` `岭回归`
 
 <font color='red'>正则化</font>的最小二乘估计。
  
- 通过加正则项的方式，放弃无偏差性，损失精度，从而得到未知数据更友好的拟合，对病态数据效果好。
- 病态矩阵：有特征线性相关性较强，也称作特征多重共线性Multicollinearity
- 传统最小二乘估计的问题，当X中含相关性较强的特征时，$\theta$系数会很大，导致x微小的变动，预测值y变化很大，导致不稳定。
- x相关性强则$\theta$大的理论解释：

> $$ min ||X\theta - y ||^2 $$
>
> $$ \theta = (X^TX)^{(-1)}X^Ty $$
> 当有特征相关性较强时，$|X^TX| 行列式接近0$ ，从而当x有微小变动时，$\theta$变化大，，导致模型不稳定。

- 正则化对样本少，特征多的情况 尤其有效

#### `Lasso Regeression`

- 对特征选择有效
- polynomial future transformation: 能捕捉特征之间的非线性关系？？？？

#### `Logistic Regeression`
在普通线性回归的公式后加了一层sigmoid函数，使得y映射到[0,1]区间平滑地变成分类问题。

- 效果跟支持向量机差不多
- 默认加入L2正则化项 （默认C=1） 
- C为正则化项系数的倒数，C大，正则化项系数越低，在train data上努力效果好。 
- C小，模型系数也会更低，甚至会损失traindata的准确率。


#### `Linear Support Vector Machine ` `线性支持向量机`
在普通线性回归的公式后加了一层sign函数，使得y映射到0/1上成为分类问题。

在所有可能的分类结果中，decision margin最大线性模型，称作线性支持向量机 LSVM. 

通过C控制正则化想的方式，调整decision margin对错判的容忍率

- decision margin: decision boundary能达到的最大宽度
- 效果性质（C)跟逻辑回归差不多
- 线性支持向量机只用到了部分trainset, 用到的部分被称作支持向量 (support vector)

#### 线性模型小结
优点：

- 简单、训练容易
- 快
- 好解释
- 能处理大数据量、稀疏数据

缺点：

- 低维特征，别的模型可能有更好的泛化效果 ？？？？
- 对于分类问题，实际问题并不一定能线性分类。

### 3. 非线性模型

#### `Kernalized Support Vector Machine`
将特征映射到高维，再用线性支持向量机

#### `Redial Basis Function Kernel` `径向基核函数`

#### `Decision Tree` `决策树`

## 特征工程

`Standardization` `标准化`

- 在Redge Regeression上做minmax标准化，效果提升显著，但在原始Least Squared Regeression上没有效果？？


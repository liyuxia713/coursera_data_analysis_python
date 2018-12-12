
# Machine Learning

<!-- TOC -->

- [Machine Learning](#machine-learning)
    - [模型评价](#模型评价)
        - [1. 分类模型评价](#1-分类模型评价)
        - [2. 回归模型评价](#2-回归模型评价)
        - [交叉验证](#交叉验证)
        - [试多个模型](#试多个模型)
    - [模型介绍](#模型介绍)
        - [1. 机器学习思想](#1-机器学习思想)
            - [`KNN`](#knn)
        - [2. 线性回归](#2-线性回归)
            - [`Least Squared Regeression` `（最小二乘估计）`](#least-squared-regeression-最小二乘估计)
        - [3. 正则化的线性回归](#3-正则化的线性回归)
            - [`Ridge Regeression` `岭回归`](#ridge-regeression-岭回归)
            - [`Lasso Regeression`](#lasso-regeression)
        - [4. 线性回归拟合非线性关系](#4-线性回归拟合非线性关系)
        - [5. 线性回归到分类](#5-线性回归到分类)
            - [`Logistic Regeression`](#logistic-regeression)
            - [`Linear Support Vector Machine ` `线性支持向量机`](#linear-support-vector-machine--线性支持向量机)
            - [`Multi-Classification`  `多分类`](#multi-classification--多分类)
        - [线性模型小结](#线性模型小结)
        - [6. 非线性模型](#6-非线性模型)
            - [`Kernalized Support Vector Machine`](#kernalized-support-vector-machine)
            - [`Decision Tree` `决策树`](#decision-tree-决策树)
        - [概率统计模型](#概率统计模型)
            - [`Naive Bayes` `朴素贝叶斯`](#naive-bayes-朴素贝叶斯)
            - [`Random Forest` `随机森林`](#random-forest-随机森林)
            - [`Gradient Boosted Decision Tree`](#gradient-boosted-decision-tree)
            - [`Neural Network` `神经网络`](#neural-network-神经网络)
            - [`Deep Learning` `深度学习`](#deep-learning-深度学习)
    - [特征工程](#特征工程)

<!-- /TOC -->

## 模型评价

### 1. 分类模型评价
准确率 F-score

### 2. 回归模型评价

`R-Squared Value`

R平方， 也称作决定系数，在线性回归模型中等于相关系数的平方， 一般理解为$1-残差平方和/样本方差$。 
介于[0,1], 越大越好。
### 交叉验证
- 5Fold
- cross validation 
- stratified cross validation (标准数据按比例分配, 避免样本部分有序的情况）
- sklearn.model_selection validation curve 画出不同参数的效果图

### 试多个模型
[ A few useful things to know about machine learning][https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf]

## 模型介绍

### 1. 机器学习思想

#### `KNN`
用N临近投票或加权平均的算法，简单计数。

- 简单，计数即可，不用模型。
- 预测效果较好
- 不能很好的抽象关系
- 不稳定（噪声影响大）

### 2. 线性回归

#### `Least Squared Regeression` `（最小二乘估计）`

最小化偏差平方和的方法

- 能很好的抽象关系
- 效果比KNN相对差
- 稳定性比KNN好

### 3. 正则化的线性回归

最小二乘估计中，如果某个特征的系数很大时，该特征微小的变化会导致结果偏差很大而效果不稳定。

- 什么时候特征系数会很大？
    - 当有特征线性相关时， $\theta$系数会很大. 
    - 这种现象称作特征多重共线性，Multicolliearity, 特征矩阵称作病态举证
    - x相关性强则$\theta$大的理论解释：

```
$$ min ||X\theta - y ||^2 $$

$$ \theta = (X^TX)^{(-1)}X^Ty $$

当有特征相关性较强时，$|X^TX|$ 行列式接近0 ，从而当x有微小变动时，$\theta$变化大，，导致模型不稳定。
```

解决思路： 在最优化函数后面加上x系数的和, 称作正则化项，从而避免系数过大。

- 通过加正则项的方式，放弃无偏差性，损失精度，从而得到未知数据更友好的拟合，对病态数据效果好。

- 正则化对样本少，特征多的情况 尤其有效
- 根据加入的正则化项的形式，有两个常见正则化方法

#### `Ridge Regeression` `岭回归`
- 正则化项为L2
- 会倾向于每个特征的系数都很小, 不会丢失特征，解释性差


#### `Lasso Regeression`
Least Absolute shrinkage and selection operator

- 正则化想为L1
- 倾向不重要的特征系数为0， 达到特征选择的效果

`为什么Lasso能做到特征选择，而Reige不能?`
http://sofasofa.io/forum_main_post.php?postid=1001156

### 4. 线性回归拟合非线性关系
polynomial future transformation
通过多项式特征的构造捕获特征之间的非线性关系
因为高阶特征容易导致过拟合，所以多项式特征通常跟正则化的线性回归，如Reige组合使用。

### 5. 线性回归到分类
思想：普通线性回归后再加一层函数，将结果映射到[0,1]区间，进而变成0/1的二分类问题
根据加的函数不同，有不同的模型。

#### `Logistic Regeression`
加的是sigmoid函数，使得y映射到[0,1]区间, 平滑地变成分类问题。
效果跟支持向量机模型差不多

也有C参数控制正则化
- 默认加入L2正则化项 （默认C=1） 
- C为正则化项系数的倒数，C大，正则化项系数越低，在train data上努力效果好。 
- C小，模型系数也会更低，甚至会损失traindata的准确率。

#### `Linear Support Vector Machine ` `线性支持向量机`
加的是sign函数，使得y映射到0/1上成为分类问题。然后再从所有可能的分类结果中，选择decision margin最大的线性模型，称作线性支持向量机 LSVM. 

- decision margin: decision boundary能达到的最大宽度
- 有分类错的情形，decision margin怎么办？
	- 通过C控制正则化项的方式，调整decision margin对错判的容忍率
- 为什么叫支持向量机？
	- 线性支持向量机只用到了部分trainset, 用到的部分被称作支持向量 (support vector)

#### `Multi-Classification`  `多分类` 
将N分类问题变成N个1 v.s. N-1的二分类问题，然后从结果中选择得分最高的

### 线性模型小结
优点：

- 简单、训练容易
- 快
- 好解释
- 能处理大数据量、稀疏数据

缺点：

- 低维特征，别的模型可能有更好的泛化效果 ？？？？
- 对于分类问题，实际关系可能很复杂，并不一定能线性分类。

### 6. 非线性模型

#### `Kernalized Support Vector Machine`
将特征映射到高维，再高维上用线性支持向量机分类。
$$ K(x, x^') = exp( -\gamma \dot ||x-x^'|| $$

参数：

- $\gamma$称作kernel width parameter, 其越小， decision margin越平滑，越大，margin 越尖，太大倾向过拟合。
- 如果$\gamma$很大，C几乎没有作用。 如果$\gamma$很小，C作用同线性支持向量机。
- 一般两个参数都是取是取中值附近，具体看你的应用场景，并测试。 $\gamma$测试在0.0001到10之间，C测试在0.1到100之间。 

- Q: After training a Radial Basis Function (RBF) kernel SVM, you decide to increase the influence of each training point and to simplify the decision surface. Which of the following would be the best choice for the next RBF SVM you train?
- A: Decrease Gamma and C
    - simplify the decision surface  (decrease gamma)

常见核函数有：

- `Redial Basis Function Kernel` `径向基核函数` RBF 
- `Polynomal Kernel` 
- `linear`

小结：

- 优点
	- 在很多数据集上表现良好。
	- 功能多样（各种核函数，甚至可以自定义）
	- 低维或高维特征都适用 
- 缺点
	- 性能差，当样本超过5万时不太实用了。
	- 特征需要标准化，且参数对效果的影响比较大。
	- 解释性差。 
	- 分类结果没有准确概率做参考。  （现在有方法可以间接的做 Platt Scaling）

#### `Decision Tree` `决策树`
通过一系列Yes or No的问题，抽象出规则，进行分类或回归。 回归问题时用叶子节点的俊辉。

容易过拟合，采用pre pruning或post pruning, 调整的参数有下面三项，通常只用一个就可以解决过拟合，建议优先有max_depth.

- max_depth
- max_leafnodes
- min_sample_leaf

小结：

- 优点：
	- 解释性非常好
	- 不用做特征预处理
	- 能处理混合类型的特征，部分连续特征，部分离散特征
- 缺点：
	- 容易过拟合得到局部最优解 （即使加入参数）
	- 可以通过决策树林的方式解决上个问题	 

### 概率统计模型
#### `Naive Bayes` `朴素贝叶斯`
是基于`贝叶斯定理`和`特征间条件独立`的`分类`算法.  Naive是指的特征间条件独立。
sklearn中有三种朴素贝叶斯:

- GaussianNB: 连续型特征上应用，如房价
- 伯努利型，特征都是二元的
- 多项式型，二元特征还要考虑权重

特点：

- 效率高
- 没有参数
- 不需要特征标准化
- 损失准确率
- 特征特别多，其他模型性能差的时候考虑使用（如文本分类中，每个word是一个特征)

可以证明，朴素贝叶斯分类器在数学上与线性模型相关，因此线性模型的许多优点和缺点也适用于朴素贝叶斯。

todo
- 条件独立
- 局部拟合

#### `Random Forest` `随机森林`
- 森林，指决策树的集合
- 随机，指每棵树随机选择样本和特征； 尽量让每棵树用到的样本和特征不同，最大程度保证树的多样性，从而避免单棵树的过拟合问题； 抽样过程是完全独立的（即有放回的抽样）
- 结果整合：分类问题，不同树的分类概率取均值，取均值最高的分类。 回归问题直接用均值。
- 参数：
	- n_estimators: 树的个数。（default=10)
	- max_features: 每棵树用的最大特征个数。 默认值效果就很好。（分类default=sqrt(N), 回归default=log(N,2), N是整体特征个数)
	- max_depth: 树的深度
	- n_jobs: 用几个核去并行计算。 （-1用所有核）
	- random_state: 随机种子，设为固定值
- 优点：
	- 在很多问题上效果较好 （如breast dataset上比前文的分类效果都好）
	- 不用做特征标准化，参数也较少。
	- 可以有不同类型的特征。
	- 可以做并行化
- 缺点：
	- 结果不好解释
	- 不适合特征特别多的场景（如文本分类） 

#### `Gradient Boosted Decision Tree`
#### `Neural Network` `神经网络`
#### `Deep Learning` `深度学习`



## 特征工程

`Standardization` `标准化`

- 在Redge Regeression上做minmax标准化，效果提升显著，但在原始Least Squared Regeression上没有效果？？


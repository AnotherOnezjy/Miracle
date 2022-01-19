# 第一章 预备知识

## 一、Python 基础

### 1. 列表推导式与条件赋值

在生成一个数字序列的时候，在 `Python` 中可以如下写出：

```python
L = []
def my_func(x):
    return 2*x
for i in range(5):
    L.append(my_func(i))
print(L)
# [0, 2, 4, 6, 8]
```

事实上可以利用列表推导式进行写法上的简化：`[* for i in *]`。其中，第一个 `*` 为映射函数，其输入为后面 `i` 指代的内容，第二个 `*` 表示迭代的对象。

```python
[my_func(i) for i in range(5)]
# [0, 2, 4, 6, 8]
```

列表表达式还支持多层嵌套，如下面的例子中第一个 `for` 为外层循环，第二个为内层循环：

```python
[m+'_'+n for m in ['a', 'b'] for n in ['c', 'd']]
# ['a_c', 'a_d', 'b_c', 'b_d']
```

除了列表推导式，另一个实用的语法糖是带有 `if` 选择的条件赋值，其形式为 `value = a if condition else b`：

```python
value = 'cat' if 2>1 else 'dog'
print(value)
# 'cat'
```

等价于如下的写法：

```python
a, b = 'cat', 'dog'
condition = 2 > 1 # 此时为True
if condition:
    value = a
else:
    value = b
```

下面举一个例子，截断列表中超过 5 的元素，即超过 5 的用 5 代替，小于 5 的保留原来的值：

```python
L = [1, 2, 3, 4, 5, 6, 7]
[i if i <= 5 else 5 for i in L]
# [1, 2, 3, 4, 5, 5, 5]
```

### 2. 匿名函数与 map 方法

有一些函数的定义具有清晰简单的映射关系，例如上面的 `my_func` 函数，这时候可以用匿名函数的方法简洁地表示：

```python
my_func = lambda x: 2*x
my_func(3) # 6
```

```python
multi_para_func = lambda a, b: a + b
multi_para_func(1, 2) 
```

但上面的用法其实违背了“匿名”的含义，事实上它往往在无需多处调用的场合进行使用，例如上面列表推导式中的例子，用户不关心函数的名字，只关心这种映射的关系：

```python
[(lambda x: 2*x)(i) for i in range(5)]
# [0, 2, 4, 6, 8]
```

对于上述的这种列表推导式的匿名函数映射，`Python` 中提供了 `map` 函数来完成，它返回的是一个 `map` 对象，需要通过 `list` 转为列表：

```python
list(map(lambda x:2*x, range(5)))
# [0, 2, 4, 6, 8]
```

对于多个输入值的函数映射，可以通过追加迭代对象实现：

```python
list(map(lambda x, y: str(x)+'_'+y, range(5), list('abcde')))
# ['0_a', '1_b', '2_c', '3_d', '4_e']
```

### 3. zip 对象与 enumerate 方法

`zip` 函数能够把多个可迭代对象打包成一个元组构成的可迭代对象，它返回了一个 `zip` 对象，通过 `tuple` ，`list` 可以得到相应的打包结果：

```python
L1, L2, L3 = list('abc'), list('def'), list('hij')
list(zip(L1, L2, L3))
# [('a', 'd', 'h'), ('b', 'e', 'i'), ('c', 'f', 'j')]
```

```python
tuple(zip(L1, L2, L3))
# (('a', 'd', 'h'), ('b', 'e', 'i'), ('c', 'f', 'j'))
```

往往会在循环迭代的时候使用到 `zip` 函数：

```python
for i, j, k in zip(L1, L2, L3):
     print(i, j, k)
# a d h
# 
# b e i
# 
# c f j
```

`enumarate` 是一种特殊的打包，它可以在迭代时绑定迭代元素的遍历序号：

```python
L = list('abcd')
for index, value in enumerate(L):
     print(index, value)
# 0 a
# 1 b
# 2 c
# 3 d
```

用 `zip` 对象也能够简单地实现这个功能：

```python
for index, value in zip(range(len(L)), L):
     print(index, value)
```

当需要对两个列表建立字典映射时，可以利用 `zip` 对象：

```python
dict(zip(L1, L2))
# {'a': 'd', 'b': 'e', 'c': 'f'}
```

既然有了压缩函数，那么 `Python` 也提供了 `*` 操作符和 `zip` 联合使用来进行解压操作：

```python
zipped = list(zip(L1, L2, L3))
print(zipped)
# [('a', 'd', 'h'), ('b', 'e', 'i'), ('c', 'f', 'j')]
list(zip(*zipped)) # 三个元组分别对应原来的列表
# [('a', 'b', 'c'), ('d', 'e', 'f'), ('h', 'i', 'j')]
```

## 二、Numpy 基础

### 1. np 数组的构造

最一般的方法是通过 `array` 来构造：

```python
import numpy as np
np.array([1, 2, 3])
# array([1, 2, 3])
```

下面讨论一些特殊数组的生成方式：<br/>

【a】等差序列：`np.linspace`，`np.arange`

```python
np.linspace(1, 5, 11)  # 起始、终止（包含）、样本个数
# array([1. , 1.4, 1.8, 2.2, 2.6, 3. , 3.4, 3.8, 4.2, 4.6, 5. ])
```

```python
np.arange(1, 5, 2)  # 起始、终止（不包含）、步长
# array([1, 3])
```

【b】特殊矩阵：`np.zeros`，`np.eye`，`np.full`

```python
np.zeros((2, 3))  # 传入元组表示各维度大小
# array([[0., 0., 0.],
# ,       [0., 0., 0.]])
```

```python
np.eye(3)  # 3 * 3 的单位矩阵
# array([[1., 0., 0.],
# ,       [0., 1., 0.],
# ,       [0., 0., 1.]])
```

```python
np.eye(3, k=1)  # 偏移主对角线 1 个单位的伪单位矩阵
# array([[0., 1., 0.],
# ,       [0., 0., 1.],
# ,       [0., 0., 0.]])
```

```python
np.full((2, 3), 10)  # 元组传入大小，10 表示填充数值
# array([[10, 10, 10],
# ,       [10, 10, 10]])
```

```python
np.full((2, 3), [1, 2, 3])  # 每行填入相同的列表
# array([[1, 2, 3],
# ,       [1, 2, 3]])
```

【c】随机矩阵：`np.random`<br/>

最常用的随机生成函数为 `rand`，`randn`，`randint`，`choice`，它们分别表示 0-1 均匀分布的随机数组、标准正态分布的随机数组、随机整数组和随机列表抽样：

```python
np.random.rand(3)  # 生成服从 0-1 均匀分布的三个随机数
# array([0.92340835, 0.20019461, 0.40755472])
```

```python
np.random.rand(3, 3)  # 注意这里传入的不是元组，每个维度大小分开输入
# array([[0.27406232, 0.15754534, 0.20274407],
# ,       [0.70255975, 0.71797703, 0.29850324],
# ,       [0.03186549, 0.54798699, 0.93826835]])
```

对于服从区间 `a`  到 `b` 上的均匀分布可以如下生成：

```python
a, b = 5, 15
(b - a) * np.random.rand(3) + a
# array([6.59370831, 8.03865138, 9.19172546])
```

一般的，可以选择已有的库函数：

```python
np.random.uniform(5, 15, 3)
# array([11.26499636, 13.12311185,  6.00774156])
```

`randn` 生成了 `N(0, I)` 的标准正态分布：

```python
np.random.randn(3)
# array([ 1.87000209,  1.19885561, -0.58802943])
```

```python
np.random.randn(2, 2)
# array([[-1.3642839 , -0.31497567],
# ,       [-1.9452492 , -3.17272882]])
```

对于服从方差为 $\sigma^2$，均值为 $\mu$ 的一元正态分布可以如下生成：

```python
sigma, mu = 2.5, 3
mu + np.random.randn(3) * sigma
# array([1.56024917, 0.22829486, 7.3764211 ])
```

同样的，也可选择从已有函数生成：

```python
np.random.normal(3, 2.5, 3)
# array([3.53517851, 5.3441269 , 3.51192744])
```

`randint` 可以指定生成随机整数的最小值最大值（不包含）和维度大小：

```python
low, high, size = 5, 15, (2, 2)  # 生成 5 到 14 的随机整数
np.random.randint(low, high, size)
# array([[ 5, 12],
# ,       [14,  9]])
```

`choice` 可以从给定的列表，以一定概率和方式抽取结果，当不指定概率时为均匀采样，默认抽取方式为有放回抽样：

```python
my_list = ['a', 'b', 'c', 'd']
np.random.choice(my_list, 2, replace=False, p=[0.1, 0.7, 0.1, 0.1])
# array(['b', 'a'], dtype='<U1')
```

```python
np.random.choice(my_list, (3, 3))
# array([['c', 'b', 'd'],
# ,       ['d', 'a', 'd'],
# ,       ['a', 'c', 'd']], dtype='<U1')
```

当返回的元素个数与原列表相同时，不放回抽样等价于使用 `permutation` 函数，即打散原列表：

```python
np.random.permutation(my_list)
# array(['c', 'a', 'd', 'b'], dtype='<U1')
```

最后，需要提到的是随机种子，它能够固定随机数的输出结果：

```python
np.random.seed(0)
np.random.rand()
# 0.5488135039273248
```

### 2. np 数组的变形与合并

【a】转置：`T`

```python
np.zeros((2, 3)).T
# array([[0., 0.],
# ,       [0., 0.],
# ,       [0., 0.]])
```

【b】合并操作：`r_`，`c_`<br/>

对于二维数组而言，`r_` 和 `c_` 分别表示上下合并和左右合并：

```python
np.r_[np.zeros((2, 3)), np.zeros((2, 3))]
# array([[0., 0., 0.],
# ,       [0., 0., 0.],
# ,       [0., 0., 0.],
# ,       [0., 0., 0.]])
```

```python
np.c_[np.zeros((2, 3)), np.zeros((2, 3))]
# array([[0., 0., 0., 0., 0., 0.],
# ,       [0., 0., 0., 0., 0., 0.]])
```

一维数组和二维数组进行合并时，应当把其视作列向量，在长度匹配的情况下只能够使用左右合并的 `c_` 操作：

```python
try:
    np.r_[np.array([0, 0]), np.zeros((2, 1))]
excepy Exception as e:
    Err_Msg = e
Err_Msg
# ValueError('all the input arrays must have same number of dimensions, but the array at index 0 has 1 dimension(s) and the array at index 1 has 2 dimension(s)')
```

```python
np.r_[np.array([0, 0]), np.zeros(2)]
# array([0., 0., 0., 0.])
```

```python
np.c_[np.array([0, 0]), np.zeros((2, 3))]
# array([[0., 0., 0., 0.],
# ,       [0., 0., 0., 0.]])
```

【c】维度变换：`reshape`<br/>

`reshape` 能够帮助用户把原数组按照新的维度重新排列。在使用 时有两种模式，分别为 `C` 模式和 `F` 模式，分别以逐行和逐列的顺序进行填充读取。

```python
target = np.arange(8).reshape(2, 4)
target
# array([[0, 1, 2, 3],
# ,       [4, 5, 6, 7]])
```

```python
target.reshape((4, 2), order='C')  # 按照行读取和填充
# array([[0, 1],
# ,       [2, 3],
# ,       [4, 5],
# ,       [6, 7]])
```

```python
target.reshape((4, 2), order='F')  # 按照列读取和填充
# array([[0, 2],
# ,       [4, 6],
# ,       [1, 3],
# ,       [5, 7]])
```

特别地，由于被调用数组地大小是确定的，`reshape` 允许有一个维度存在空缺，此时只需填充 -1 即可：

```python
target.reshape((4, -1))
# array([[0, 1],
# ,       [2, 3],
# ,       [4, 5],
# ,       [6, 7]])
```

下面将 `n * 1` 大小的数组转为一维数组的操作是经常使用的：

```python
target = np.ones((3, 1))
target
# array([[1.],
# ,       [1.],
# ,       [1.]])
target.reshape(-1)
# array([1., 1., 1.])
```

### 3. np 数组的切片与索引

数组的切片模式支持使用 `slice` 类型的 `start:end:step` 切片，还可以直接传入列表指定某个维度的索引进行切片：

```python
target = np.arange(9).reshape(3, 3)
target
# array([[0, 1, 2],
# ,       [3, 4, 5],
# ,       [6, 7, 8]])
target[:-1, [0, 2]]
# array([[0, 2],
# ,       [3, 5]])
```

此外，还可以利用 `np.ix_` 在对应维度上使用布尔索引，但此时不能使用 `slice` 切片：

```python
target[np.ix_([True, False, True], [True, False, True])]
# array([[0, 2],
# ,       [6, 8]])
target[np.ix_([1, 2], [True, False, True])]
# array([[3, 5],
# ,       [6, 8]])
```

当数组维度变为一维时，可以直接进行布尔索引，而无需 `np.ix_`：

```python
new = target.reshape(-1)
new[new % 2 == 0]
# array([0, 2, 4, 6, 8])
```

### 4. 常用函数

为了简单起见，这里假设下述函数输入的数组都是一维的。<br/>

【a】`where`<br/>

`where` 是一种条件函数，可以指定满足条件与不满足条件位置对应的填充值：

```python
a = np.array([-1, 1, -1, 0])
np.where(a > 0, a, 5)  # 对应位置为 True 时填充 a 对应元素，否则填充 5
# array([5, 1, 5, 5])
```

【b】`nonzero`，`argmax`，`argmin`<br/>

这三个函数返回的都是索引，`nonzero` 返回非零数的索引，`argmax`、`argmin` 分别返回最大和最小数的索引：

```python
a = np.array([-2, -5, 0, 1, 3, -1])
np.nonzero(a)
# (array([0, 1, 3, 4, 5], dtype=int64),)
a.argmax()  # 4
a.argmin()  # 1
```

【c】`any`，`all`<br/>

`any` 指当序列至少**存在一个** `True` 或非零元素时返回 `True`，否则返回 `False`；`all` 指当序列元素**全为** `True` 或非零元素时返回 `True`，否则返回 `False`。

```python
a = np.array([0, 1])
a.any()  # True
a.all()  # False
```

【d】`cumprod`，`cumsum`，`diff`<br/>

`cumprod`，`cumsum` 分别表示累乘和累加函数，返回同长度的数组，`diff` 表示和前一个元素作差，由于第一个元素为缺失值，因此在默认参数情况下，返回长度是原数组长度减 1。

```python
a = np.array([1, 2, 3])
a.cumprod()
# array([1, 2, 6], dtype=int32)
a.cumsum()
# array([1, 3, 6], dtype=int32)
np.diff(a)
# array([1, 1])
```

【e】统计函数<br/>

常用的统计函数包括 `max`，`min`，`mean`，`median`，`std`，`var`，`sum`，`quantile`，其中分位数计算是全局方法，因此不能通过 `array.quantile` 的方法调用：

```python
target = np.arange(5)
target
# array([0, 1, 2, 3, 4])
target.max()
# 4
np.quantile(target, 0.5)  # 0.5 分位数
# 2.0
```

但是对于含有缺失值的数组，它们返回的结果也是缺失值，如果需要略过缺失值，必须使用 `nan*` 类型的函数，上述的几个统计函数都有对应的 `nan*` 函数。

```python
target = np.array([1, 2, np.nan])
target
# array([ 1.,  2., nan])
target.max()
# nan
np.nanmax(target)
# 2.0
np.nanquantile(target, 0.5)
# 1.5
```

对于协方差和相关系数分别可以利用 `cov`，`corroef` 来计算：

```python
target1 = np.array([1, 3, 5, 9])
target2 = np.array([1, 5, 3, -9])
np.cov(target1, target2)
# array([[ 11.66666667, -16.66666667],
# ,       [-16.66666667,  38.66666667]])
np.corroef(target1, target2)
# array([[ 1.        , -0.78470603],
# ,       [-0.78470603,  1.        ]])
```

最后，需要说明二维 `Numpy` 数组中统计函数的 `axis` 参数，它能够进行某一个维度下的统计特征计算，当 `axis=0` 时结果为列的统计指标，当 `axis=1` 时结果为行的统计指标：

```python
target = np.arange(1, 10).reshape(3, -1)
target
# array([[1, 2, 3],
# ,       [4, 5, 6],
# ,       [7, 8, 9]])
target.sum(0)
# array([12, 15, 18])
target.sum(1)
# array([ 6, 15, 24])
```

### 5. 广播机制

广播机制用于处理两个不同维度数组之间的操作，这里只讨论不超过两维的数组广播机制。<br/>

【a】标量和数组的操作<br/>

当一个标量和数组进行运算时，标量会自动把大小扩充为数组大小，之后进行逐元素操作：

```python
res = 3 * np.ones((2, 2)) + 1
res
# array([[4., 4.],
# ,       [4., 4.]])
res = 1 / res
res
# array([[0.25, 0.25],
# ,       [0.25, 0.25]])
```

【b】二维数组之间的操作<br/>

当两个数组维度完全一致时，使用对应元素的操作，否则会报错，除非其中的某个数组的维度是 $m \times 1$ 或 $1 \times n$，那么会扩充其具有 1 的维度为另一个数组对应维度的大小。例如，$1 \times 2$ 数组和 $3 \times 2$ 数组做逐元素运算时会把第一个数组扩充为 $3 \times 2$，扩充时的对应数值进行赋值。但是，需要注意的是，如果第一个数组的维度是 $1 \times 3$，那么由于第二维上的大小不匹配且不为 1，此时报错。

```python
res = np.ones((3, 2))
res
# array([[1., 1.],
# ,       [1., 1.],
# ,       [1., 1.]])
res * np.array([[2, 3]])  # 第二个数组扩充第一维度为 3
# array([[2., 3.],
# ,       [2., 3.],
# ,       [2., 3.]])
res * np.array([[2], [3], [4]])  # 第二个数组扩充第二维度为 2
# array([[2., 2.],
# ,       [3., 3.],
# ,       [4., 4.]])
res * np.array([[2]])  # 等价于两次扩充，第二个数组两个维度分别扩充为 3 和 2
# array([[2., 2.],
# ,       [2., 2.],
# ,       [2., 2.]])
```

【c】一维数组与二维数组的操作<br/>

当一维数组 $A_k$ 与二维数组 $B_{m, n}$ 操作时，等价于把一维数组视作 $A_{i, k}$ 的二维数组，使用的广播法则与 【b】 中一致。当 $k \neq n$ 且 $k, n$ 都不是 1 时报错。

```python
np.ones(3) + np.ones((2, 3))
# array([[2., 2., 2.],
# ,       [2., 2., 2.]])
np.ones(3) + np.ones((2, 1))
# array([[2., 2., 2.],
# ,       [2., 2., 2.]])
np.ones(1) + np.ones((2, 3))
# array([[2., 2., 2.],
# ,       [2., 2., 2.]])
```

### 6. 向量与矩阵的运算

【a】向量内积：`dot`
$$
a \cdot b = \sum_{i} {a_ib_i}
$$

```python
a = np.array([1, 2, 3])
b = np.array([1, 3, 5])
a.dot(b)  # 22
```

【b】向量范数和矩阵范数：`np.linalg.norm`<br/>

在矩阵范数的计算中，最重要的是 `ord` 参数，可选值如下：

|  ord  | norm for matrices            | norm for vectors             |
| :---: | ---------------------------- | ---------------------------- |
| None  | Frobenius norm               | 2-norm                       |
| 'fro' | Frobenius norm               | /                            |
| 'nuc' | nuclear norm                 | /                            |
|  inf  | max(sum(abs(x), axis=1))     | max(abs(x))                  |
| -inf  | min(sum(abs(x), axis=1))     | min(abs(x))                  |
|   0   | /                            | sum(x != 0)                  |
|   1   | max(sum(abs(x), axis=0))     | as below                     |
|  -1   | min(sum(abs(x), axis=0))     | as below                     |
|   2   | 2-norm (largest sing. value) | as below                     |
|  -2   | smallest singular value      | as below                     |
| other | /                            | `sum(abs(x)**ord)**(1./ord)` |

```python
matrix_target = np.arange(4).reshape(-1, 2)
matrix_target
# array([[0, 1],
# ,       [2, 3]])
np.linalg.norm(matrix_target, 'fro')
# 3.7416573867739413
np.linalg.norm(matrix_target, np.inf)
# 5.0
np.linalg.norm(matrix_target, 2)
# 3.702459173643833
vector_target = np.arange(4)
vector_target
# array([0, 1, 2, 3])
np.linalg.norm(vector_target, np.inf)
# 3.0
np.linalg.norm(vector_target, 2)
# 3.7416573867739413
np.linalg.norm(vector_target, 3)
# 3.3019272488946263
```

【c】矩阵乘法：`@`
$$
[A_{m \times p}B_{p \times n}] = \sum_{k=1}^{p} {A_{i, k}B_{k, j}}
$$

```python
a = np.arange(4).reshape(-1, 2)
a
# array([[0, 1],
# ,       [2, 3]])
b = np.arange(-4, 0).reshape(-1, 2)
b
# array([[-4, -3],
# ,       [-2, -1]])
a @ b
# array([[ -2,  -1],
# ,       [-14,  -9]])
```

## 三、练习

### Ex1：利用列表推导式实现矩阵乘法

一般的矩阵乘法根据公式，可以由三重循环写出，请将其改写为**列表推导式**的形式。

- stem

```python
M1 = np.random.rand(2, 3)
M2 = np.random.rand(3, 4)
res = np.empty((M1.shape[0], M2.shape[1]))
for i in range(M1.shape[0]):
    for j in range(M2.shape[1]):
        item = 0
        for k in range(M1.shape[1]):
            item += M1[k][k] * M2[k][j]
        res[i][j] = item
(np.abs((M1@M2 - res) < 1e-15)).all()  # 排除数值误差
# True
```

- code

```python
res = [[sum([M1[i][k] * M2[k][j] for k in range(M1.shape[1])])
        for j in range(M2.shape[1])] for i in range(M1.shape[0])]
```

### Ex2：更新矩阵

- stem

设矩阵 $A_{m \times n}$，现在对 $A$ 中的每一个元素进行更新生成矩阵 $B$，更新方法是 $B_{ij} = A_{ij} \sum_{k=1}^{n} \frac {1}{A_{ik}}$，例如下面的矩阵为 $A$，则 $B_{2, 2} = 5 \times ( \frac{1}{4} + \frac{1}{5} + \frac{1}{6}) = \frac{37}{12}$，请利用 `Numpy` 高效实现。
$$
A = 
\left[
\begin{array}{l}
	1 & 2 & 3 \\
	4 & 5 & 6 \\
	7 & 8 & 9
\end{array}
\right]
$$

- code

```python
A = np.arange(1, 10).reshape(3, -1)
B = A * (1 / A).sum(1).reshape(-1, 1)
```

### Ex3：卡方统计量

- stem

设矩阵 $A_{m \times n}$，记 $B_{ij} = \frac {(\sum_{i=1}^{m} {A_{ij}}) \times (\sum_{j=1}^{n} {A_{ij}})}{\sum_{i=1}^{m}\sum_{j=1}^{n} {A_{ij}}}$，定义卡方值如下：
$$
\chi^2 = \sum_{i=1}^{m}\sum_{j=1}^{n}\frac {(A_{ij}-B_{ij})^2}{B_{ij}}
$$
请利用 `Numpy` 对给定的矩阵 $A$ 计算 $\chi^2$。

```python
np.random.seed(0)
A = np.random.randint(10, 20, (8, 5))
```

- code

```python
# 构造 B 矩阵
B = A.sum(0) * A.sum(1).reshape(-1, 1) / A.sum()
# 计算卡方值
chi_square = ((A - B) ** 2 / B).sum()
```

### Ex4：改进矩阵计算的性能

- stem

设 $Z$ 为 $m \times n$ 的矩阵，$B$ 和 $U$ 分别是 $m \times p$ 和 $p \times n$ 的矩阵，$B_i$ 为 $B$ 的第 $i$ 行，$U_j$ 为 $U$ 的第 $j$ 列，下面定义 $R = \sum_{i=1}^{m} \sum_{j=1}^{n} {\left|| B_i - U_j \right||^2_2 Z_{ij}}$，其中 $\left|| a \right||_2^2$ 表示向量 $a$ 的分量平方和 $\sum_i a_i^2$。<br/>

现有某人根据如下给定的样例数据计算 $R$ 的值，请充分利用 `Numpy` 中的函数，基于此问题改进这段代码的性能。

```python
np.random.seed(0)
m, n, p = 100, 80, 50
B = np.random.randint(0, 2, (m, p))
U = np.random.randint(0, 2, (p, n))
Z = np.random.randint(0, 2, (m, n))
def solution(B=B, U=U, Z=Z):
    L_res = []
    for i in range(m):
        for j in range(n):
            norm_value = ((B[i]-U[:,j])**2).sum()
            L_res.append(norm_value*Z[i][j])
    return sum(L_res)
solution(B, U, Z)  # 100566
```

- think

令 $Y_{ij} = \left|| B_i - U_j \right||_2^2$，则 $R = \sum_{i=1}^{m} \sum_{j=1}^{n} {Y_{ij}Z_{ij}}$，这在 `Numpy` 中可以用逐元素的乘法后求和实现，因此问题转化为了如何构造 $Y$ 矩阵。我们对计算式作如下变形：
$$
\begin{align*}
Y_{ij} &= \left|| B_i - U_j\right||_2^2 \\
	   &= \sum_{k=1}^p {(B_{ik} - U_{kj})^2} \\
	   &= \sum_{k=1}^p {B_{ik}^2} + \sum_{k=1}^p {U_{ik}^2} - 2\sum_{k=1}^p {B_{ik}U_{kj}}
\end{align*}
$$
从上式可以看出，第一、第二项分别为 $B$ 的行平方和与 $U$ 的列平方和，第三项是两倍的内积。因此，$Y$ 矩阵可以写为三个部分，第一个部分是 $m \times n$ 的全 1 矩阵每行乘以 $B$ 对应行的行平方和，第二个部分是相同大小的全 1 矩阵每列乘以 $U$ 对应列的列平方和，第三个部分恰为 $B$ 矩阵与 $U$ 矩阵乘积的两倍。结果如下：

- code

```python
def solution_improved(B=B, U=U, Z=Z):
    Y = (B**2).sum(1).reshape(-1, 1) + (U**2).sum(0) - 2 * B @ U
    return (Y * Z).sum()

solution_improved(B, U, Z)  # 100566
```

性能对比：

- ```python
  %timeit -n 30 solution(B, U, Z)
  # 36.2 ms ± 129 µs per loop (mean ± std. dev. of 7 runs, 30 loops each)
  ```

- ```python
  %timeit -n 30 solution_improved(B, U, Z)
  # 269 µs ± 13.7 µs per loop (mean ± std. dev. of 7 runs, 30 loops each)
  ```

### Ex5：连续整数的最大长度

- stem

输入一个整数的 `Numpy` 数组，返回其中严格递增连续整数子数组的最大长度，正向是指递增方向。例如，输入 [1, 2, 5, 6, 7]，[5, 6, 7] 为具有最大长度的连续整数子数组，因此输出 3；输入 [3, 2, 1, 2, 3, 4, 6]，[1, 2, 3, 4] 为具有最大长度的连续整数子数组，因此输出 4。请充分利用 `Numpy` 的内置函数完成。（提示：考虑使用 `nonzero`，`diff` 函数）

- code

```python
f = lambda x:np.diff(np.nonzero(np.r_[1,np.diff(x)!=1,1])).max()
```

# 第二章 Pandas 基础

```python
import numpy as np
import pandas as pd
```

在开始学习前，请保证 `pandas` 的版本号不低于如下所示的版本，否则请务必升级！请确认已经安装了 `xlrd`，`xlwt`，`openpyxl` 这三个包，其中 `xlrd` 版本不得高于 `2.0.0`。

```python
pd.__version__
# '1.2.0'
```

## 一、文件的读取和写入

### 1. 文件读取

`pandas` 可以读取的文件格式有很多，这里主要介绍读取 `csv`，`excel`，`txt` 文件。

```python
df_csv = pd.read_csv('...')
df_txt = pd.read_txt('...')
df_excel = pd.read_excel('...')
```

这里有一些常用的公共参数，`header=None` 表示第一行不作为列名，`index_col` 表示把某一列或几列作为索引，索引的内容将会在第三章进行详述，`usecols` 表示读取列的集合，默认读取所有的列，`parse_dates` 表示需要转化为时间的列，关于时间序列的有关内容将在第十章讲解，`nrows` 表示读取的数据行数。上面这些参数在上述的三个函数里都可以使用。

```python
pd.read_table('....txt', header=None)
pd.read_csv('....csv', index_col=['col1', 'col2'])
pd.read_table('....txt', usecols=['col1', 'col2'])
pd.read_csv('....csv', parse_dates=['col5'])
pd.read_excel('....xlsx', nrows=2)
```

在读取 `txt` 文件时，经常遇到分隔符非空格的情况，`read_table` 有一个分割参数 `sep`，它使得用户可以自定义分割符号，进行 `txt` 数据的读取。例如，下面读取的表以 `||||` 为分割。直接读取的结果显然不是理想的，这时可以使用 `sep`，同时需要指定引擎为 `python`：

```python
pd.read_table('....txt', sep=' \|\|\|\| ', engine='python')
```

【WARNING】`sep` 是正则参数<br/>

在使用 `read_table` 的时候需要注意，参数 `sep` 中使用的是正则表达式，因此需要对 `|` 进行转义变成 `\|`，否则无法读取到正确的结果。有关正则表达式的基本内容可以参考第八章或者其它相关资料。

### 2. 数据写入

一般在数据写入中，最常用的操作是把 `index` 设置为 `False`，特别当索引没有特殊意义的时候，这样的行为能把索引在保存的时候去除。

```python
df_csv.to_csv('....csv', index=False)
df_excel.to_excel('....xlsx', index=False)
```

`pandas` 中没有定义 `to_table` 函数，但是 `to_csv` 可以保存为 `txt` 文件，并且允许自定义分隔符，常用制表符 `\t` 分割：

```python
df_txt.to_csv('....txt', sep='\t', index=False)
```

如果想要把表格快速转换为 `markdown` 和 `latex` 语言，可以使用 `to_markdown` 和 `to_latex` 函数，此处都需要安装 `tabulate` 包。

```python
print(df_csv.to_markdown())
print(df_csv.to_latex())
```

## 二、基本数据结构

`pandas` 中具有两种基本的数据存储结构，存储一维 `values` 的 `Series` 和存储二维 `values` 的 `DataFrame`，在这两种结构上定义了很多的属性和方法。

### 1. Series

`Series` 一般由四个部分组成，分别是序列的值 `data`、索引 `index`、存储类型 `dtype`、序列的名字 `name`。其中，索引也可以指定它的名字，默认为空。

```python
s = pd.Series(data = [100, 'a', {'dic1':5}],
              index = pd.Index(['id1', 20, 'third'], name='my_idx'),
              dtype = 'object',
              name = 'my_name')
```

【NOTE】`object` 类型<br/>

`object` 代表了一种混合类型，正如上面的例子中存储了整数、字符串以及 `Python` 的字典数据结构。此外，目前 `pandas` 把纯字符串序列也默认认为是一种 `object` 类型的序列，但它也可以用 `string` 类型存储，文本序列的内容会在第八章中讨论。<br/>

对于这些属性，可以通过 `.` 的方式来获取：

```python
s.values
# array([100, 'a', {'dic1': 5}], dtype=object)
s.index
# Index(['id1', 20, 'third'], dtype='object', name='my_idx')
s.dtype
# dtype('O')
s.name
# 'my_name'
```

利用 `.shape` 可以获取序列的长度：

```python
s.shape
# (3,)
```

索引是 `pandas` 中最重要的概念之一，它将在第三章中被详细地讨论。如果想要取出单个索引对应的值，可以通过 `[index_item]` 可以取出。

### 2. DataFrame

`DataFrame` 在 `Series` 的基础上增加了列索引，一个数据框可以由二维的 `data` 与行列索引来构造：

```python
data = [[1, 'a', 1.2], [2, 'b', 2.2], [3, 'c', 3.2]]
df = pd.DataFrame(data = data,
                  index = ['row_%d'%i for i in range(3)],
                  columns=['col_0', 'col_1', 'col_2'])
```

但一般而言，更多的时候会采用从列索引名到数据的映射来构造数据框，同时再加上行索引：

```python
df = pd.DataFrame(data = {'col_0': [1,2,3],
                          'col_1':list('abc'),
                          'col_2': [1.2, 2.2, 3.2]},
                  index = ['row_%d'%i for i in range(3)])
```

由于这种映射关系，在 `DataFrame` 中可以用 `[col_name]` 与 `[col_list]` 来取出相应的列与由多个列组成的表，结果分别为 `Series` 和 `DataFrame`：

```python
df['col_0']
df['col_0', 'col_1']
```

与 `Series` 类似，在数据框中同样可以取出相应的属性：

```python
df.values
df.index
df.columns
df.dtypes  # 返回的是值为相应列数据类型的 Series
df.shape
```

通过 `.T` 可以把 `DataFrame` 进行转置：

```python
df.T
```

## 三、常用基本函数

为了进行举例说明，在接下来的部分和其余章节都将会使用一份 `learn_pandas.csv` 的虚拟数据集，它记录了四所学校学生的体测个人信息。

```python
df = pd.read_csv('....csv')
df.columns
# Index(['School', 'Grade', 'Name', 'Gender', 'Height', 'Weight', 'Transfer',
# ,       'Test_Number', 'Test_Date', 'Time_Record'],
# ,      dtype='object')
```

上述列名依次代表学校、年级、姓名、性别、身高、体重、是否为转系生、体测场次、测试时间、1000 米成绩，本章只需使用其中的前七列。

```python
df = df[df.colomns[:7]]
```

### 1. 汇总函数

`head`，`tail` 函数分别表示返回表或者序列的前 `n` 行和后 `n` 行，其中 `n` 默认为 5：

```python
df.head(2)
```

|      | School                        | Grade    | Name           | Gender | Height | Weight | Transfer |
| ---- | ----------------------------- | -------- | -------------- | ------ | ------ | ------ | -------- |
| 0    | Shanghai Jiao Tong University | Freshman | Gaopeng Yang   | Female | 158.9  | 46.0   | N        |
| 1    | Peking University             | Freshman | Changqiang You | Male   | 166.5  | 70.0   | N        |

```python
df.tail(3)
```

|      | School                        | Grade     | Name           | Gender | Height | Weight | Transfer |
| ---- | ----------------------------- | --------- | -------------- | ------ | ------ | ------ | -------- |
| 197  | Shanghai Jiao Tong University | Senior    | Chengqiang Chu | Female | 153.9  | 45.0   | N        |
| 198  | Shanghai Jiao Tong University | Senior    | Chengmei Shen  | Male   | 175.3  | 71.0   | N        |
| 199  | Tsinghua University           | Sophomore | Chunpeng Lv    | Male   | 155.7  | 51.0   | N        |

`info`，`describe` 分别返回表的信息概况和表中数值列对应的主要统计量：

```python
df.info()
'''
<class 'pandas.core.frame.DataFrame'>

RangeIndex: 200 entries, 0 to 199

Data columns (total 7 columns):

 #   Column    Non-Null Count  Dtype  

---  ------    --------------  -----  

 0   School    200 non-null    object 

 1   Grade     200 non-null    object 

 2   Name      200 non-null    object 

 3   Gender    200 non-null    object 

 4   Height    183 non-null    float64

 5   Weight    189 non-null    float64

 6   Transfer  188 non-null    object 

dtypes: float64(2), object(5)

memory usage: 11.1+ KB
'''
```

```python
df.describe()
```

|       |   Height   |   Weight   |
| :---: | :--------: | :--------: |
| count | 183.000000 | 189.000000 |
| mean  | 163.218033 | 55.015873  |
|  std  |  8.608879  | 12.824294  |
|  min  | 145.400000 | 34.000000  |
|  25%  | 157.150000 | 46.000000  |
|  50%  | 161.900000 | 51.000000  |
|  75%  | 167.500000 | 65.000000  |
|  max  | 193.900000 | 89.000000  |

【NOTE】更全面的数据汇总

`info`，`describe` 只能实现较少信息的展示，如果想要对一份数据集进行全面且有效的观察，特别是在列较多的情况下，推荐使用 `pandas-profiling` 包，它将在第十一章被再次提到。
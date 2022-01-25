# 第一章 线性规划

## 1.1 线性规划问题

### 1.1.1 线性规划的实例与定义

线性规划问题是在一组线性约束条件的限制下，求一线性目标函数最大或最小值的问题。

-   目标函数
-   约束条件（subject to，s.t.）

由于目标函数及约束条件均为线性函数，故称为线性规划问题。

### 1.1.2 线性规划问题的解的概念

一般线性问题的（数学）标准型为
$$
max z = \sum_{j=1}^{n} {c_jx_j}
$$

$$
s.t. \left\{
\begin{aligned}
\sum_{j=1}^{n} {a_{ij}x_j = b_i}，i &= 1, 2, \cdots, m \\
x_j \geq 0, j &= 1, 2, \cdots, n。
\end{aligned}
\right.
$$
式中：$b_i \geq 0,i = 1, 2, \cdots, m$。

- **可行解**

  满足约束条件式（2）的解 $\boldsymbol x = [x_1, x_2, \cdots, x_n]^T$，称为线性规划问题的**可行解**，而使目标函数式（1）达到最大值的可行解称为**最优解**。

- **可行域**

  所有可行解构成的集合称为问题的可行域，记为 ***R***。

### 1.1.3 线性规划的 Matlab 标准形式及软件求解

为了避免线性规划形式多样性带来的不便，`Matlab` 中规定线性规划的标准形式
$$
\underset{x}{min} = \boldsymbol{f}^T\boldsymbol x, \\
s.t. \left\{
\begin{aligned}
& \boldsymbol A \cdot \boldsymbol x \leq \boldsymbol b, \\
& Aeq \cdot \boldsymbol x = beq, \\
& lb \leq \boldsymbol x \leq ub
\end{aligned}
\right.
$$
式中：$f,\space x,\space b,\space beq,\space lb,\space ub$ 为列向量，其中 $f$ 称为价值向量，$b$ 称为资源向量；$\boldsymbol A$, $Aeq$ 为矩阵。`Matlab` 中求解线性规划的命令为：<br/>

[x, fval] = linprog(f, A, b)<br/>

[x, fval] = linprog(f, A, b, Aeq, beq)<br/>

[x, fval] = linprog(f, A, b, Aeq, beq, lb, ub)<br/>

式中：x 返回决策向量的取值；fval 返回目标函数的最优值；f 为价值向量；A 和 b 对应线性不等式约束；Aeq 和 beq 对应线性等式约束；lb 和 ub 分别对应决策向量的下界向量和上界向量。例如，线性规划
$$
\underset{x}{max} \space \boldsymbol c^T \boldsymbol x, \\
s.t. \space \boldsymbol A \boldsymbol x \geq \boldsymbol b
$$
的 `Matlab` 标准型为
$$
\underset{x}{min} - \boldsymbol c^T \boldsymbol x, \\
s.t. \space -\boldsymbol A \boldsymbol x \leq -\boldsymbol b
$$

一般的，$\boldsymbol c$ 是一个列向量。各个矩阵定义时不必化为标准型的格式，只需在传参时保证满足要求即可。 

### 1.1.4 可以转化为线性规划的问题

很多看起来不是线性规划的问题，也可以通过变换转化为线性规划的问题来解决。<br/>

【例】数学规划问题：
$$
min \abs{x_1} + \abs{x_2} + \cdots + \abs{x_n}, \\
s.t. \space \boldsymbol A \boldsymbol x \leq \boldsymbol b
$$
式中：$\boldsymbol x = [x_1, \cdots, x_n]^T$；$\boldsymbol A$ 和 $\boldsymbol b$ 为相应维数的矩阵和向量。<br/>

 注意到：对于任意的 $x_i$，存在 $u_i,v_i \geq 0$ 满足
$$
x_i = u_i - v_i, \space \abs{x_i} = u_i + v_i
$$
取 $u_i = \frac {\abs{x_i}+x_i}{2},v_i = \frac{\abs{x_i}-x_i}{2}$ 就可以满足上面的条件。<br/>

记 $\boldsymbol u = [u_1, \cdots, u_n]^T,\boldsymbol v = [v_1, \cdots, v_n]^T$，上面的问题化为
$$
min \sum_{i=1}^n {(u_i + v_i)}, \\
s.t. \left\{
\begin{aligned}
& \boldsymbol A(\boldsymbol u - \boldsymbol v) \leq b,\\
& \boldsymbol u, \boldsymbol v \geq 0
\end{aligned}
\right.
$$
式中：$\boldsymbol u \geq 0$ 为向量 $\boldsymbol u$ 的每个分量大于等于 0。<br/>

进一步把模型改写为
$$
min \sum_{i=1}^n {(u_i + v_i)},\\
s.t. \left\{
\begin{aligned}
& \left[\boldsymbol A, -\boldsymbol A\right]\begin{bmatrix}u\\v\end{bmatrix} \leq \boldsymbol b,\\
& \boldsymbol u, \boldsymbol v \geq 0。
\end{aligned}
\right.
$$

注意，最后还需要通过 $\boldsymbol x = \boldsymbol u - \boldsymbol v$ 还原。

## 1.2 投资的收益和风险

待续……



# Reference

[1] 司守奎，孙兆亮，数学建模算法与应用（第二版），北京：国防工业出版社，2017.4
[2] 司守奎等，数学建模算法与应用习题解答（第二版），北京：国防工业出版社，2017.4

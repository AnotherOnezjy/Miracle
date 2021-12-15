## 几种触发器的功能表和激励表

### 1 与非门基本 R-S 触发器

-   示意图
    <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/1639563482695.png"></img>

-   功能表和激励表
    <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/1639563132691.png"></img>

-   方程

1. 次态方程

$$
Q^{n+1} = \overline{S} + RQ
$$

2. 约束方程

$$
R + S = 1
$$

-   说明

1. 当与非门构成的基本 R-S 触发器的同一输入端连续出现多个负脉冲信号时，仅第一个使触发器状态发生改变
2. 可以用来消除毛刺
3. **下降沿触发**

### 2 或非门基本 R-S 触发器

-   示意图
    <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/1639563682658.png"></img>

-   功能表和激励表
    <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/1639563734296.png"></img>

-   方程

1. 次态方程

$$
Q^{n+1} = S + \overline{R}Q
$$

2. 约束方程

$$
R \cdot S = 0
$$

### 3 钟控 R-S 触发器

-   示意图
    <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/1639563953694.png"></img>

-   功能表和激励表
    <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/1639564023153.png"></img>

-   方程

1. 次态方程

$$
Q^{n+1} = S + \overline{R}Q
$$

2. 约束方程

$$
R \cdot S = 0
$$

### 4 钟控 D 触发器

-   示意图
    <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/1639564195517.png"></img>

-   功能表和激励表
    <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/1639564307137.png"></img>

-   次态方程

$$
Q^{n+1} = D
$$

### 5 钟控 J-K 触发器

-   示意图
    <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/1639564498681.png"></img>

-   功能表和激励表
    <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/1639564575454.png"></img>

-   次态方程

$$
Q^{n+1} = J\overline{Q} + \overline{K}Q
$$

### 6 钟控 T 触发器

-   示意图
    <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/1639564720075.png"></img>

-   功能表和激励表
    <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/1639564765360.png"></img>

-   次态方程

$$
Q^{n+1} = T \oplus Q
$$

### 未完待续……

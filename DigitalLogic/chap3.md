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
3. 置 d 时以 R 为准

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

- 说明
  1. **上升沿触发**
  2. 置 d 时以 S 为准

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

- 说明
  1. 同或非门基本 R-S 触发器
  2. 置 d 时以 S 为准

### 4 钟控 D 触发器

-   示意图
    <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/1639564195517.png"></img>

-   功能表和激励表
    <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/1639564307137.png"></img>

-   次态方程

$$
Q^{n+1} = D
$$

- 说明

  - 即置 D
  - 由于钟控 D 触发器在时钟脉冲作用后的次态和输入 D 的值一致，故有时又称为**锁存器**。

### 5 钟控 J-K 触发器

-   示意图
    <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/1639564498681.png"></img>

-   功能表和激励表（功能表有误，置 1 和置 0 应互换）
    <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/1639564575454.png"></img>

-   次态方程

$$
Q^{n+1} = J\overline{Q} + \overline{K}Q
$$

- 说明
  1. 00不变，11翻转
  2. 置 d 时以 J 为准

### 6 钟控 T 触发器

-   示意图
    <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/1639564720075.png"></img>

-   功能表和激励表
    <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/1639564765360.png"></img>

-   次态方程

$$
Q^{n+1} = T \oplus Q
$$

- 说明

  - 即将钟控 J-K 触发器的 J、K 合并为了一个端 T，所以次态方程 $ Q^{n+1} = T\overline{Q} + \overline{T}Q = T \oplus Q$
  - 钟控 T 触发器只要有时钟脉冲作用，触发器状态就翻转，相当于一位二进制计数器，故又称为**计数触发器**。

## 空翻现象

### 空翻的原因

- 时钟脉冲作用期间，输入信号发生变化，触发器状态会跟着变化
- 时钟宽度控制不够，输入的多次变化得到完全响应，使得一个时钟脉冲作用期间触发器多次翻转

## 主从钟控触发器

### 主从 R-S 触发器

- 示意图

  <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/%E4%B8%BB%E4%BB%8ER-S%E8%A7%A6%E5%8F%91%E5%99%A8%E7%A4%BA%E6%84%8F%E5%9B%BE-2021-12-2316:44:32.png"></img>

- 作用时间

  - CP = 1 **主应从定**

  - CP = 0 **从应主定**

- 总结
  - 前沿采样，后沿定局
  - 状态变化在时钟脉冲的**后沿**
  - 无空翻现象

- 功能

  <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/%E4%B8%BB%E4%BB%8ER-S%E8%A7%A6%E5%8F%91%E5%99%A8%E5%8A%9F%E8%83%BD%E5%9B%BE-2021-12-2316:40:47.png"></img>

与钟控 R-S 触发器一致

### 主从 J-K 触发器

- 示意图

  <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/%E4%B8%BB%E4%BB%8EJ-K%E8%A7%A6%E5%8F%91%E5%99%A8%E7%A4%BA%E6%84%8F%E5%9B%BE-2021-12-2316:48:58.png"></img>

- 功能

  与 J-K 触发器一致

- 缺点

  存在**一次翻转**问题，即一个时钟脉冲周期内只能采样第一次变化

  <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/%E4%B8%BB%E4%BB%8EJ-K%E8%A7%A6%E5%8F%91%E5%99%A8%E4%B8%80%E6%AC%A1%E7%BF%BB%E8%BD%AC-2021-12-2317:08:00.png"></img>

## 维持-阻塞钟控触发器

### 维持-阻塞 D 触发器

- 示意图

  <img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/%E7%BB%B4%E6%8C%81%E9%98%BB%E5%A1%9ED%E8%A7%A6%E5%8F%91%E5%99%A8%E7%A4%BA%E6%84%8F%E5%9B%BE-2021-12-2317:12:41.png"></img>

- 功能

  - CP = 0

    Q 不变

  - CP = 1

    - Q = D
    - 无空翻

  - 实际中使用的维持-阻塞 D 触发器有时具有几个 D 输入端，此时各输入之间是**相与**的关系。例如，当有 3 个输入端 $D_1$、$D_2$ 和 $D_3$ 时，次态方程是：
    $$
    Q^{n+1} = D_1D_2D_3
    $$
    

- 优点

  - 前沿采样
  - 在上升沿过后的时钟脉冲周期，D的值可以随意改变
  - 无空翻
    - 脉冲的边沿作用
    - 维护阻塞作用

## 未完待续……

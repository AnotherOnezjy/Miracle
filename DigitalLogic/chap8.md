# 第八章 可编程逻辑器件

## 一、PLD 概述

### 数字系统中常用的大规模集成电路

- 非用户定制电路
- 全用户定制电路
- 半用户定制电路

可编程逻辑器件（简称 PLD）属于**半用户**定制电路

- 可编程只读存储器（PROM）：由一个“与”阵列和一个“或”阵列组成，**“与”阵列固定，“或”阵列可编程**
- 可编程逻辑阵列（PLA）：由一个“与”阵列和一个“或”阵列组成，**“与”阵列和“或”阵列都可编程**
- 可编程阵列逻辑（PAL）：**“与”阵列可编程，“或”阵列固定**

## 二、低密度可编程逻辑器件

根据 PLD 中阵列和输出结构的不同，目前常用的低密度 PLD 有4种主要类型：

- 可编程只读存储器 PROM

- 可编程逻辑阵列 PLA

- 可编程阵列逻辑 PAL

- 通用阵列逻辑 GAL

### PROM

#### 逻辑结构

<div align=center><img
    src="https://gitee.com/Miraclezjy/utoolspic/raw/master/PROM%E7%9A%84%E9%80%BB%E8%BE%91%E7%BB%93%E6%9E%84-2021-12-2510:51:47.png">
</div>

- **容量**：n 位地址输入，m 位数据输出，存储容量为 $2^n \times m$ 位

画阵列图时，将 PROM 中的每个与门和或门都简化成一根线

#### PROM应用举例

改变“或”阵列上连接点的数量和位置，就可以在输出端形成由输入变量“最小项之和”表示的任何一种逻辑函数

##### 设计过程

• 根据逻辑要求列出真值表

• 根据逻辑函数值确定对PROM“或”阵列进行编程的代码，画出相应的阵列图

### PLA

#### 逻辑结构

- 由一个“与”阵列和一个“或”阵列构成，“与”阵列和“或”阵列都是可编程的 

- 在 PLA 中，n 个输入变量的“与”阵列通过编程提供需要的 P 个“与”项，“或”阵列通过编程形成“与-或”函数式 

- 由 PLA 实现的函数式是**最简“与-或”表达式**

#### 容量

n-p-m 表示法

- n：输入变量数
- p：与项数
- m：输出端数

## 三 、高密度可编程逻辑器件



## 四、在系统编程技术简介


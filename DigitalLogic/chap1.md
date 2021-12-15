## 几种常用的编码

### 8421 码

-   `0000` 到 `1001`，九个数

### 2421 码

-   `0000` 到 `0100`，前 5 个数同 8421 码
-   `1011` 到 `1111`，后 5 个数不同

### 余 3 码

-   即 8421 码加上 3

## 奇偶校验

### 奇校验

对校验位 P，有：
$$ P = \overline{b_1 \oplus b_2 \oplus b_3 \oplus b_4 \oplus b_5 \oplus b_6 \oplus b_7 \oplus b_8} $$

### 偶校验

对校验位 P，有：
$$ P = b_1 \oplus b_2 \oplus b_3 \oplus b_4 \oplus b_5 \oplus b_6 \oplus b_7 \oplus b_8 $$

### 奇偶校验的特点

-   只需要 **1 位**校验码
-   只具有**发现错误**的能力，不具备对错误定位继而纠正错误的能力
-   只具有发现一串二进制代码中同时出现**奇数个代码出错**的能力

## 循环冗余校验码（CRC）

？？？

## 格雷码（Gray Code）

特点：任意两个相邻的数，其格雷码仅有一位不同。

### 格雷码转换

<img src="https://gitee.com/Miraclezjy/utoolspic/raw/master/1639530995464.png"></img>
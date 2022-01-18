# 第一章 数据类型及数组创建

## 一、常量

### numpy.nan

- 表示空值

  nan = NaN = NAN

【例】两个`numpy.nan`是不相等的

```python
import numpy as np

print(np.nan == np.nan)  # False
print(np.nan != np.nan)  # True
```

- numpy.isnan(x, *args, **kwargs)：Test element-wise for NaN and return result as a boolean array.

【例】

```python
import numpy as np

x = np.array([1, 1, 8, np.nan, 10])
print(x)
# [ 1.  1.  8. nan 10.]

y = np.isnan(x)
print(y)
# [False False False  True False]

z = np.count_nonzero(y)
print(z)  # 1
```

###  numpy.inf

- 表示正无穷大

  Inf = inf = infty = Infinity = PINF

  NINF = -inf

【例】

```python
import numpy as np

print(np.inf) # inf
print(np.NINF) # -inf
```

### numpy.pi

- 表示圆周率

【例】

```python
import numpy as np

print(np.pi)  # 3.141592653589793
```

### numpy.e

- 表示自然常数

【例】

```python
import numpy as np

print(np.e)  # 2.718281828459045
```

## 二、数据类型

### 常见数据类型

numpy 基本类型

| 类型                              | 备注           | 说明       |
| :-------------------------------- | :------------- | ---------- |
| bool_ = bool8                     | 8位            | 布尔类型   |
| int8 = byte                       | 8位            | 整型       |
| int16 = short                     | 16位           | 整型       |
| int32 = intc                      | 32位           | 整型       |
| int_ = int64 = long = int0 = intp | 64位           | 整型       |
| uint8 = ubyte                     | 8位            | 无符号整型 |
| uint16 = ushort                   | 16位           | 无符号整型 |
| uint32 = uintc                    | 32位           | 无符号整型 |
| uint64 = uintp = uint0 = uint     | 64位           | 无符号整型 |
| float16 = half                    | 16位           | 浮点型     |
| float32 = single                  | 32位           | 浮点型     |
| float_ = float64 = double         | 64位           | 浮点型     |
| str_ = unicode_ = str0 = unicode  | Unicode 字符串 |            |
| datetime64                        | 日期时间类型   |            |

### 创建数据类型

numpy 的数值类型实际上是 dtype 对象的实例。

```python
class dtype(object):
    def __init__(self, obj, align=False, copy=False):
        pass
```

每个内建类型都有一个唯一定义它的字符代码，如下：

| 字符 | 对应类型               | 备注                   |
| :--: | ---------------------- | ---------------------- |
|  b   | boolean                | 'b1'                   |
|  i   | signed integer         | 'i1', 'i2', 'i4', 'i8' |
|  u   | unsigned integer       | 'u1', 'u2' ,'u4' ,'u8' |
|  f   | floating-point         | 'f2', 'f4', 'f8'       |
|  c   | complex floating-point |                        |
|  m   | timedelta64            | 表示两个时间之间的间隔 |
|  M   | datetime64             | 日期时间类型           |
|  O   | object                 |                        |
|  S   | (byte-)string          | S3表示长度为3的字符串  |
|  U   | Unicode                | Unicode 字符串         |
|  V   | void                   |                        |

【例】

```python
import numpy as np

a = np.dtype('b1')
print(a.type)  # <class 'numpy.bool_'>
print(a.itemsize)  # 1

a = np.dtype('i1')
print(a.type)  # <class 'numpy.int8'>
print(a.itemsize)  # 1
a = np.dtype('i2')
print(a.type)  # <class 'numpy.int16'>
print(a.itemsize)  # 2
a = np.dtype('i4')
print(a.type)  # <class 'numpy.int32'>
print(a.itemsize)  # 4
a = np.dtype('i8')
print(a.type)  # <class 'numpy.int64'>
print(a.itemsize)  # 8

a = np.dtype('u1')
print(a.type)  # <class 'numpy.uint8'>
print(a.itemsize)  # 1
a = np.dtype('u2')
print(a.type)  # <class 'numpy.uint16'>
print(a.itemsize)  # 2
a = np.dtype('u4')
print(a.type)  # <class 'numpy.uint32'>
print(a.itemsize)  # 4
a = np.dtype('u8')
print(a.type)  # <class 'numpy.uint64'>
print(a.itemsize)  # 8

a = np.dtype('f2')
print(a.type)  # <class 'numpy.float16'>
print(a.itemsize)  # 2
a = np.dtype('f4')
print(a.type)  # <class 'numpy.float32'>
print(a.itemsize)  # 4
a = np.dtype('f8')
print(a.type)  # <class 'numpy.float64'>
print(a.itemsize)  # 8

a = np.dtype('S')
print(a.type)  # <class 'numpy.bytes_'>
print(a.itemsize)  # 0
a = np.dtype('S3')
print(a.type)  # <class 'numpy.bytes_'>
print(a.itemsize)  # 3

a = np.dtype('U3')
print(a.type)  # <class 'numpy.str_'>
print(a.itemsize)  # 12
```

### 数据类型信息

Python 中的浮点数通常是64位浮点数，几乎等同于 `np.float64`。

Numpy 和 Python 整数类型的行为在整数溢出方面存在显著差异，与 Numpy 不同，Python 的 `int` 是灵活的。这意味着 Python 整数可以扩展以容纳任何整数并且不会溢出。

Machines limits for integer types.

```python
class iiinfo(object):
    def __init__(self, int_type):
        pass
    def min(self):
        pass
    def max(self):
        pass
```

【例】

```python
import numpy as np

ii16 = np.iinfo(np.int16)
print(ii16.min)  # -32768
print(ii16.max)  # 32767

ii32 = np.iinfo(np.int32)
print(ii32.min)  # -2147483648
print(ii32.max)  # 2147483647
```

Machine limits for floating point types.

```python
class finfo(object):
    def _init(self, dtype):
```

【例】

```python
import numpy as np

ff16 = np.finfo(np.float16)
print(ff16.bits)  # 16
print(ff16.min)  # -65500.0
print(ff16.max)  # 65500.0
print(ff16.eps)  # 0.000977

ff32 = np.finfo(np.float32)
print(ff32.bits)  # 32
print(ff32.min)  # -3.4028235e+38
print(ff32.max)  # 3.4028235e+38
print(ff32.eps)  # 1.1920929e-07
```

## 三、时间日期和时间增量

### datetime64基础

在 numpy 中，我们很方便的将字符串转换成时间日期类型 `datetime64`(`datetime64`已被 python 包含的日期时间库所占用)。`datetime64`是带单位的日期时间类型，其单位如下：

| 日期单位 | 代码含义 | 时间单位 | 代码含义 |
| :------: | :------: | :------: | :------: |
|    Y     |    年    |    h     |   小时   |
|    M     |    月    |    m     |   分钟   |
|    W     |    周    |    s     |    秒    |
|    D     |    天    |    ms    |   毫秒   |
|    -     |    -     |    us    |   微秒   |
|    -     |    -     |    ns    |   纳秒   |
|    -     |    -     |    ps    |   皮秒   |
|    -     |    -     |    fs    |   飞秒   |
|    -     |    -     |    as    |  阿托秒  |

【例】从字符串创建 datetime64 类型时，默认情况下，numpy 会根据字符串自动选择对应的单位。

```python
import numpy as np

a = np.datetime64('2020-03-01')
print(a, a.dtype)  # 2020-03-01 datetime64[D]

a = np.datetime64('2020-03')
print(a, a.dtype)  # 2020-03 datetime64[M]

a = np.datetime64('2020-03-08 20:00:05')
print(a, a.dtype)  # 2020-03-08T20:00:05 datetime64[s]

a = np.datetime64('2020-03-08 20:00')
print(a, a.dtype)  # 2020-03-08T20:00 datetime64[m]

a = np.datetime64('2020-03-08 20')
print(a, a.dtype)  # 2020-03-08T20 datetime64[h]
```

【例】从字符串创建 datetime64 类型时，可以强制指定使用的单位。

```python
import numpy as np

a = np.datetime64('2020-03', 'D')
print(a, a.dtype)  # 2020-03-01 datetime64[D]

a = np.datetime64('2020-03', 'Y')
print(a, a.dtype)  # 2020 datetime64[Y]

print(np.datetime64('2020-03') == np.datetime64('2020-03-01'))  # True
print(np.datetime64('2020-03') == np.datetime64('2020-03-02'))  # False
```

由这个例子可以看出，2020-03 和 2020-03-01 表示的是同一个时间。事实上，如果两个 datetime64 对象具有不同的单位，它们可能仍然代表相同的时刻。并且从较大的单位（如月份）转换为较小的单位（如天数）是安全的。

【例】从字符串创建 datetime64 数组时，如果单位不统一，则一律转化成其中最小的单位。

```python
import numpy as np

a = np.array(['2020-03', '2020-03-08', '2020-03-08 20:00'], dtype='datetime64')
print(a, a.dtype)
# ['2020-03-01T00:00' '2020-03-08T00:00' '2020-03-08T20:00'] datetime64[m]
```

【例】使用 `arange()` 创建 datetime64 数组，用于生成日期范围。

```python
import numpy as np

a = np.arange('2020-08-01', '2020-08-10', dtype=np.datetime64)
print(a)
# ['2020-08-01' '2020-08-02' '2020-08-03' '2020-08-04' '2020-08-05' '2020-08-06' '2020-08-07' '2020-08-08' '2020-08-09']
print(a.dtype)  # datetime64[D]

a = np.arange('2020-08-01 20:00', '2020-08-10', dtype=np.datetime64)
print(a)
# ['2020-08-01T20:00' '2020-08-01T20:01' '2020-08-01T20:02' ...
#  '2020-08-09T23:57' '2020-08-09T23:58' '2020-08-09T23:59']
print(a.dtype)  # datetime64[m]

a = np.arange('2020-05', '2020-12', dtype=np.datetime64)
print(a)
# ['2020-05' '2020-06' '2020-07' '2020-08' '2020-09' '2020-10' '2020-11']
print(a.dtype)  # datetime64[M]
```

### datetime64 和 timedelta64 运算

【例】timedelta64 表示两个 datetime64 之间的差。timedelta64 也是带单位的，并且和相减运算中的两个 datetime64 中的较小的单位保持一致。

```python
import numpy as np

a = np.datetime64('2020-03-08') - np.datetime64('2020-03-07')
b = np.datetime64('2020-03-08') - np.datetime64('2020-03-07 08:00')
c = np.datetime64('2020-03-08') - np.datetime64('2020-03-07 23:00', 'D')

print(a, a.dtype)  # 1 days timedelta64[D]
print(b, b.dtype)  # 960 minutes timedelta64[m]
print(c, c.dtype)  # 1 days timedelta64[D]

a = np.datetime64('2020-03') + np.timedelta64(20, 'D')
b = np.datetime64('2020-06-15 00:00') + np.timedelta64(12, 'h')
print(a, a.dtype)  # 2020-03-21 datetime64[D]
print(b, b.dtype)  # 2020-06-15T12:00 datetime64[m]
```

【例】生成 timedelta64 时，年(‘Y’)和月(‘M’)这两个单位无法和其他单位进行运算（一年的天数，一个月的小时数不确定）。

```python
import numpy as np

a = np.timedelta64(1, 'Y')
b = np.timedelta64(a, 'M')
print(a)  # 1 years
print(b)  # 12 months

c = np.timedelta64(1, 'h')
d = np.timedelta64(c, 'm')
print(c)  # 1 hours
print(d)  # 60 minutes

print(np.timedelta64(a, 'D'))
# TypeError: Cannot cast NumPy timedelta64 scalar from metadata [Y] to [D] according to the rule 'same_kind'

print(np.timedelta64(b, 'D'))
# TypeError: Cannot cast NumPy timedelta64 scalar from metadata [M] to [D] according to the rule 'same_kind'
```

【例】timedelta64 的运算

```python
import numpy as np

a = np.timedelta64(1, 'Y')
b = np.timedelta64(6, 'M')
c = np.timedelta64(1, 'W')
d = np.timedelta64(1, 'D')
e = np.timedelta64(10, 'D')

print(a)  # 1 years
print(b)  # 6 months
print(a + b)  # 18 months
print(a - b)  # 6 months
print(2 * a)  # 2 years
print(a / b)  # 2.0
print(c / d)  # 7.0
print(c % e)  # 7 days
```

【例】numpy.datetime64 与 datetime.datetime 相互转换

```python
import numpy as np
import datetime

dt = datetime.datetime(year=2020, month=6, day=1, hour=20, minute=5, second=30)
dt64 = np.datetime64(dt, 's')
print(dt64, dt64.dtype)
# 2020-06-01T20:05:30 datetime64[s]

dt2 = dt64.astype(datetime.datetime)
print(dt2, type(dt2))
# 2020-06-01 20:05:30 <class 'datetime.datetime'>
```

### datetime64 的应用

为了允许在只有一周中某些日子有效的上下文中使用日期时间，Numpy 包含一组“busday”（工作日）功能。

- numpy.busday_offset(dates, offsets, roll=‘raise’,weekmask=‘1111100’, holiday=None, busdaycal=None, out=None)

  First adjusts the date to fall on a valid day according to the roll rule, then applies offsets to the given dates counted in valid days.

参数 `roll`:{‘raise’, ‘nat’, ‘forward’, ‘following’, ‘backward’, ‘preceding’, ‘modifiedfollowing’, ‘modifiedpreceding’}

- ‘raise’ means to raise an exception for an invalid day.
- ‘nat’ means to return a NaT(not-a-time) for an invalid day.
- ‘forward’ and ‘following’ mean to take the first day later in time.
- ‘backward’ and ‘preceding’ mean to take the first valid day earilier in time.

【例】将指定的偏移量应用于工作日，单位天（‘D’）。计算下一个工作日，如果当前日期为非工作日，默认报错。可以指定 `forward` 或 `backward` 规则来避免报错。（一个是向前取第一个有效的工作日，一个是向后取第一个有效的工作日）

```python
import numpy as np

# 2020-07-10 星期五
a = np.busday_offset('2020-07-10', offsets=1)
print(a)  # 2020-07-13

# a = np.busday_offset('2020-07-11', offsets=1)
print(a)  # ValueError: Non-business day date in busday_offset

a = np.busday_offset('2020-07-11', offsets=0, roll='forward')
b = np.busday_offset('2020-07-11', offsets=0, roll='backward')
print(a)  # 2020-07-13
print(b)  # 2020-07-10

a = np.busday_offset('2020-07-11', offsets=1, roll='forward')
b = np.busday_offset('2020-07-11', offsets=1, roll='backward')
print(a)  # 2020-07-14
print(b)  # 2020-07-13
```

可以指定偏移量为 0 来获取当前日期向前或向后最近的工作日，当然，如果当前日期本身就是工作日，则直接返回当前日期。

- numpy.is_busday(dates, weekmask=‘1111100’, holidays=None, busdaycal=None, out=None) Calculates which of the given dates are valid days, and which are not.

【例】返回指定日期是否是工作日

```python
import numpy as np

# 2020-07-10 星期五
a = np.is_busday('2020-07-10')
b = np.is_busday('2020-07-11')
print(a)  # True
print(b)  # False
```

【例】统计一个 `datetime64[D]` 数组中的工作日天数。

```python
import numpy as np

# 2020-7-10 星期五
begindates = np.datetime64('2020-07-10')
enddates = np.datetime64('2020-07-20')
a = np.arange(begindates, enddates, dtype='datetime64')
b = np.count_nonzero(np.is_busday(a))
print(a)
# ['2020-07-10' '2020-07-11' '2020-07-12' '2020-07-13' '2020-07-14' '2020-07-15' '2020-07-16' '2020-07-17' '2020-07-18' '2020-07-19']
print(b)  # 6
```

【例】自定义周掩码值，即指定一周中哪些星期是工作日。

```python
import numpy as np

# 2020-07-10 星期五
a = np.is_busday('2020-07-10', weekmask=[1, 1, 1, 1, 1, 0, 0])
b = np.is_busday('2020-07-10', weekmask=[1, 1, 1, 1, 0, 0, 1])
print(a)  # True
print(b)  # False
```

- numpy.busday_count(begindates, enddates, weekmasks=‘1111100’, holidays=[], busdaycal=None, out=None) Counts the number of valid days between begindates and enddates, not including the day of enddates.

【例】返回两个日期之间的工作日数量。

```python
import numpy as np

# 2020-07-10 星期五
begindates = np.datetime64('2020-07-10')
enddates = np.datetime64('2020-07-20')
a = np.busday_count(begindates, enddates)
b = np.busday_count(enddates, begindates)
print(a)  # 6
print(b)  # -6
```

## 四、数组的创建

numpy 提供的最重要的数据结构是 `ndarray`，它是 python 中 `list` 的扩展。

### 1. 依据现有数据来创建 ndarray

1. 通过 `array()` 函数进行创建

   ```python
   def array(p_object, dtype=None, copy=True, order='K', subok=False, ndmin=0):
   ```

【例】

```python
import numpy as np

# 创建一维数组
a = np.array([0, 1, 2, 3, 4])
b = np.array([0, 1, 2, 3, 4])
print(a, type(a))
# [0 1 2 3 4] <class 'numpy.ndarray'>
print(b, type(b))
# [0 1 2 3 4] <class 'numpy.ndarray'>

# 创建二维数组
c = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
print(c, type(c))
# [[11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]
#  [26 27 28 29 30]
#  [31 32 33 34 35]] <class 'numpy.ndarray'>

# 创建三维数组
d = np.array([[(1.5, 2, 3), (4, 5, 6)],
              [(3, 2, 1), (4, 5, 6)]])
print(d, type(d))
# [[[1.5 2.  3. ]
#   [4.  5.  6. ]]
#
#  [[3.  2.  1. ]
#   [4.  5.  6. ]]] <class 'numpy.ndarray'>
```

2. 通过 `asarray()` 函数进行创建

`array()` 和 `asarray()` 都可以将结构数据转化为 ndarray，但是 `array()` 和 `asarray()` 主要区别就是 `array()` 仍然会 copy 出一个副本，占用新的内存，但不改变 dtype 时 `asarray()` 不会。

```python
def asarray(a, dtype=None, order=None):
    return array(a, dtype, copy=False, order=order)
```

【例】`array()` 和 `asarray()` 都可以将结构数据转化为 ndarray

```python
import numpy as np

x = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
y = np.array(x)
z = np.asarray(x)
x[1][2] = 2
print(x, type(x))
# [[1, 1, 1], [1, 1, 2], [1, 1, 1]] <class 'list'>

print(y, type(y))
# [[1 1 1]
#  [1 1 1]
#  [1 1 1]] <class 'numpy.ndarray'>

print(z, type(z))
# [[1 1 1]
#  [1 1 1]
#  [1 1 1]] <class 'numpy.ndarray'>
```

【例】`array()` 和 `asarray()` 的区别：当数据源是 `ndarray` 时，`array()` 仍然会 copy 出一个副本，占用新的内存，但不改变 dtype 时 `asarray()` 不会。

```python
import numpy as np

x = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
y = np.array(x)
z = np.asarray(x)
w = np.asarray(x, dtype=np.int)
x[1][2] = 2
print(x, type(x), x.dtype)
# [[1 1 1]
#  [1 1 2]
#  [1 1 1]] <class 'numpy.ndarray'> int32

print(y, type(y), y.dtype)
# [[1 1 1]
#  [1 1 1]
#  [1 1 1]] <class 'numpy.ndarray'> int32

print(z, type(z), z.dtype)
# [[1 1 1]
#  [1 1 2]
#  [1 1 1]] <class 'numpy.ndarray'> int32

print(w, type(w), w.dtype)
# [[1 1 1]
#  [1 1 2]
#  [1 1 1]] <class 'numpy.ndarray'> int32
```

【例】更改为较大的 dtype 时，其大小必须是 array 的最后一个 axis 的总大小（以字节为单位）的除数

```python
import numpy as np

x = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
print(x, x.dtype)
# [[1 1 1]
#  [1 1 1]
#  [1 1 1]] int32
x.dtype = np.float
# ValueError: When changing to a larger dtype, its size must be a divisor of the total size in bytes of the last axis of the array.
```

3. 通过 `fromfunction()` 函数进行创建

给函数绘图的时候可能会用到 `fromfunction()` ，该函数可从函数中创建数组。

```python
def fromfunction(function, shape, **kwargs):
```

【例】通过在每个坐标上执行一个函数来构造数组。

```python
import numpy as np


def f(x, y):
    return 10 * x + y


x = np.fromfunction(f, (5, 4), dtype=int)
print(x)
# [[ 0  1  2  3]
#  [10 11 12 13]
#  [20 21 22 23]
#  [30 31 32 33]
#  [40 41 42 43]]

x = np.fromfunction(lambda i, j: i == j, (3, 3), dtype=int)
print(x)
# [[ True False False]
#  [False  True False]
#  [False False  True]]

x = np.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)
print(x)
# [[0 1 2]
#  [1 2 3]
#  [2 3 4]]
```

### 2. 依据 ones 和 zeros 填充方式

在机器学习任务中经常做的一件事就是初始化常数，需要用常数值或随机值来创建一个固定大小的矩阵。

#### (a) 零数组

- `zeros()` 函数：返回给定形状和类型的零数组；
- `zeros_like()` 函数：返回与给定数组形状和类型相同的零数组。

```python
def zeros(shape, dtype=None, order='C'):
    ...

def zeros_like(a, dtype=None, order='K', subok=True, shape=None):
    ...
```

【例】

```python
import numpy as np

x = np.zeros(5)
print(x) # [0. 0. 0. 0. 0.]
x = np.zeros([2, 3])
print(x)
# [[0. 0. 0.]
#  [0. 0. 0.]]

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.zeros_like(x)
print(y)
# [[0 0 0]
#  [0 0 0]]
```

#### (b) 1 数组

- `ones()` 函数：返回给定形状和类型的 1 数组；
- `ones_like()` 函数：返回与给定数组形状和类型相同的 1 数组。

```python
def ones(shape, dtype=None, order='C'):
    ...
    
def ones_like(a, dtype=None, order='K', subok=True, shape=None):
    ...
```

#### (c) 空数组

- `empty()` 函数：返回一个空数组，数组元素为随机数；
- `empty_like` 函数：返回与给定数组具有相同形状和类型的新数组。

```python
def empty(shape, dtype=None, order='C'):
    ...
    
def empty_like(prototype, dtype=None, order='K', subok=None, shape=None):
    ...
```

#### (d) 单位数组

- `eye()` 函数：返回一个对角线上为 1，其它地方为零的单位数组；
- `identity()` 函数：返回一个方的单位数组。

```python
def eye(N, M=None, k=0, dtype=float, order='C'):
    ...
    
def identity(n, dtype=None):
    ...
```

【例】

```python
import numpy as np

x = np.eye(4)
print(x)
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]

x = np.eye(2, 3)
print(x)
# [[1. 0. 0.]
#  [0. 1. 0.]]

x = np.identity(4)
print(x)
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]
```

#### (e) 对角数组

- `diag()` 函数：提取对角线或构造对角数组。

```python
def diag(v, k=0):
    ...
```

【例】

```python
import numpy as np

x = np.arange(9).reshape((3, 3))
print(x)

print(np.diag(x))
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]
print(np.diag(x, k=1))  # [1 5]
print(np.diag(x, k=-1))  # [3 7]

v = [1, 3, 5, 7]
x = np.diag(v)
print(x)
# [[1 0 0 0]
#  [0 3 0 0]
#  [0 0 5 0]
#  [0 0 0 7]]
```

#### (f) 常数数组

- `full()` 函数：返回一个常数数组；
- `full_like` 函数：返回与给定数组具有相同形状和类型的常数数组。

```python
def full(shape, fill_value, dtype=None, order='C'):
    ...
    
def full_like(a, fill_value, order='K', sudok=True, shape=None)
```

【例】

```python
import numpy as np

x = np.full((2,), 7)
print(x)
# [7 7]

x = np.full(2, 7)
print(x)
# [7 7]

x = np.full((2, 7), 7)
print(x)
# [[7 7 7 7 7 7 7]
#  [7 7 7 7 7 7 7]]

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.full_like(x, 7)
print(y)
# [[7 7 7]
#  [7 7 7]]
```

### 3. 利用数值范围来创建 ndarray

- `arange()` 函数：返回给定间隔内的均匀间隔的值；
- `linspace()` 函数：返回指定间隔内的等间隔数字；
- `logspace()` 函数：返回数以对数刻度均匀分布；
- `numpy.random.rand()`：返回一个由 $[0, 1)$ 内的随机数组成的数组。

```python
def arange([start,] stop[, step,], dtype=None):
    ...

def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    ...

def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    ...

def rand(d0, d1, ..., dn):
    ...
```

【例】

```python
import numpy as np

x = np.arange(5)
print(x)  # [0 1 2 3 4]

x = np.arange(3, 7, 2)
print(x)  # [3 5]

x = np.linspace(start=0, stop=2, num=9)
print(x)  # [0.   0.25 0.5  0.75 1.   1.25 1.5  1.75 2.  ]

x = np.logspace(0, 1, 5)
print(np.around(x, 2))
# [ 1.    1.78  3.16  5.62 10.  ]

"""
np.around 返回四舍五入之后的值，可指定精度
around(a, decimals=0, out=None)
a 输入数组
decimals 要舍入的小数位数，默认值为 0。如果为负，整数将四舍五入到小数点左侧的位置
"""

x = np.linspace(start=0, stop=1, num=5)
x = [10 ** i for i in x]
print(np.round(x, 2))
# [ 1.    1.78  3.16  5.62 10.  ]

x = np.random.random([2, 3])
print(x)
# [[0.37457211 0.56728249 0.46809528]
#  [0.68499324 0.47909432 0.01194955]]
```

### 4. 结构数组的创建

结构数组，首先需要定义结构，然后利用 `np.array()` 来创建数组，其参数 `dtype` 为定义的结构。

#### (a) 利用字典来定义结构

【例】

```python
import numpy as np

personType = np.dtype({
    'names': ['name', 'age', 'weight'],
    'formats': ['U30', 'i8', 'f8']})

a = np.array([('Liming', 24, 63.9), ('Mike', 15, 67.), ('Jan', 34, 45.8)],
             dtype=personType)
print(a, type(a))
# [('Liming', 24, 63.9) ('Mike', 15, 67. ) ('Jan', 34, 45.8)]
# <class 'numpy.ndarray'>
```

#### (b) 利用包含多个元组的列表来定义结构

【例】

```python
import numpy as np

personType = np.dtype([('name', 'U30'), ('age', 'i8'), ('weight', 'f8')])
a = np.array([('Liming', 24, 63.9), ('Mike', 15, 67.), ('Jan', 34, 45.8)],
             dtype=personType)
print(a, type(a))
# [('Liming', 24, 63.9) ('Mike', 15, 67. ) ('Jan', 34, 45.8)]
# <class 'numpy.ndarray'>

# 结构数组的取值方式和一般数组差不多，可以通过下标取得元素：
print(a[0])
# ('Liming', 24, 63.9)

print(a[-2:])
# [('Mike', 15, 67. ) ('Jan', 34, 45.8)]

# 我们可以使用字段名作为下标获取对应的值
print(a['name'])
# ['Liming' 'Mike' 'Jan']
print(a['age'])
# [24 15 34]
print(a['weight'])
# [63.9 67.  45.8]
```

------

**数组的属性**

在使用 numpy 时，你会想知道数组的某些信息。numpy 中有许多便捷的方法，可以给我们想要的信息。

- `numpy.ndarray.ndim`：返回数组的维数（轴的个数），也称为秩。一维数组的秩为 1，二维数组的秩为 2，以此类推。
- `numpy.ndarray.shape`：表示数组的维度，返回一个元组，这个元组的长度就是维度的数目，即 ndim 属性（秩）。
- `numpy.ndarray.dtype`：ndarray 对象的元素类型。
- `numpy.ndarray.itemsize`：以字节的形式返回数组中每一个元素的大小。

```python
class ndarray(object):
    shape = property(lambda self: object(), lambda self, v: None, lambda self: None)
    dtype = property(lambda self: object(), lambda self, v: None, lambda self: None)
    size = property(lambda self: object(), lambda self, v: None, lambda self: None)
    ndim = property(lambda self: object(), lambda self, v: None, lambda self: None)
    itemsize = property(lambda self: object(), lambda self, v: None, lambda self: None)
```

【例】

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a.shape)  # (5,)
print(a.dtype)  # int32
print(a.size)  # 5
print(a.ndim)  # 1
print(a.itemsize)  # 4

b = np.array([[1, 2, 3], [4, 5, 6.0]])
print(b.shape)  # (2, 3)
print(b.dtype)  # float64
print(b.size)  # 6
print(b.ndim)  # 2
print(b.itemsize)  # 8
```

在 `ndarray` 中所有元素必须是同一类型，否则会自动向下转换，`int -> float -> str`。

【例】

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a)  # [1 2 3 4 5]
b = np.array([1, 2, 3, 4, '5'])
print(b)  # ['1' '2' '3' '4' '5']
c = np.array([1, 2, 3, 4, 5.0])
print(c)  # [1. 2. 3. 4. 5.]
```

# 第二章 索引

## 索引、切片与迭代

### 副本与视图

在 Numpy 中，尤其是在做数组运算或数组操作时，返回结果不是数组的**副本**就是**视图**。在 Numpy 中，所有赋值运算不会为数组和数组中的任何元素创建副本。

- `numpy.ndarray.copy()` 函数创建一个副本。对副本数据进行修改，不会影响到原始数据，它们物理内存不在同一位置。

【例】

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = x
y[0] = -1  # 不使用 copy，修改的是原数据
print(x)
# [-1  2  3  4  5  6  7  8]
print(y)
# [-1  2  3  4  5  6  7  8]

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = x.copy()
y[0] = -1  # 使用了 copy，修改的是副本数据，不影响原数据
print(x)
# [1 2 3 4 5 6 7 8]
print(y)
# [-1  2  3  4  5  6  7  8]
```

【例】数组切片操作返回的对象是原数组的视图。

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = x
y[::2, :3:2] = -1
print(x)
# [[-1 12 -1 14 15]
#  [16 17 18 19 20]
#  [-1 22 -1 24 25]
#  [26 27 28 29 30]
#  [-1 32 -1 34 35]]
print(y)
# [[-1 12 -1 14 15]
#  [16 17 18 19 20]
#  [-1 22 -1 24 25]
#  [26 27 28 29 30]
#  [-1 32 -1 34 35]]

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = x.copy()
y[::2, :3:2] = -1
print(x)
# [[11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]
#  [26 27 28 29 30]
#  [31 32 33 34 35]]
print(y)
# [[-1 12 -1 14 15]
#  [16 17 18 19 20]
#  [-1 22 -1 24 25]
#  [26 27 28 29 30]
#  [-1 32 -1 34 35]]
```

### 索引与切片

数组索引机制指的是用方括号加序号的形式引用单个数组元素，它的用处很多，比如抽取元素，选取数组的几个元素，甚至为其赋一个新值。

#### 整数索引

【例】要获取数组的单个元素，指定元素的索引即可。

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print(x[2])  # 3

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
print(x[2])  # [21 22 23 24 25]
print(x[2][1])  # 22
print(x[2, 1])  # 22
```

#### 切片索引

切片操作是指抽取数组的一部分元素生成新数组。对 python **列表**进行切片操作得到的数组是原数组的**副本**，而对 Numpy 数据进行切片操作得到的数组则是指向相同缓冲区的**视图**。<br/>

如果想抽取（或查看）数组的一部分，必须使用切片语法，也就是采用 `[start:stop:step]` 的方式。各个位置的缺省值情况是：`start` 默认为 0，`stop` 默认为数组的最大索引值，`step` 默认为 1，即抽取所有元素而不再考虑间隔。<br/>

【例】对一维数组的切片

```python
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print(x[0:2])  # [1 2]
# 用下标 0 ~ 5，以 2 为步长选取数组
print(x[1:5:2])  # [2 4]
print(x[2:])  # [3 4 5 6 7 8]
print(x[:2])  # [1 2]
print(x[-2:])  # [7 8]
print(x[:-2])  # [1 2 3 4 5 6]
print(x[:])  # [1 2 3 4 5 6 7 8]
# 利用负数下标翻转数组
print(x[::-1])  # [8 7 6 5 4 3 2 1]
```

【例】对二维数组的切片

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
print(x[0:2])
# [[11 12 13 14 15]
#  [16 17 18 19 20]]

print(x[1:5:2])
# [[16 17 18 19 20]
#  [26 27 28 29 30]]

print(x[2:])
# [[21 22 23 24 25]
#  [26 27 28 29 30]
#  [31 32 33 34 35]]

print(x[:2])
# [[11 12 13 14 15]
#  [16 17 18 19 20]]

print(x[-2:])
# [[26 27 28 29 30]
#  [31 32 33 34 35]]

print(x[:-2])
# [[11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]]

print(x[:])
# [[11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]
#  [26 27 28 29 30]
#  [31 32 33 34 35]]

print(x[2, :])  # [21 22 23 24 25]
print(x[:, 2])  # [13 18 23 28 33]
print(x[0, 1:4])  # [12 13 14]
print(x[1:4, 0])  # [16 21 26]
print(x[1:3, 2:4])
# [[18 19]
#  [23 24]]

print(x[:, :])
# [[11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]
#  [26 27 28 29 30]
#  [31 32 33 34 35]]

print(x[::2, ::2])
# [[11 13 15]
#  [21 23 25]
#  [31 33 35]]

print(x[::-1, :])
# [[31 32 33 34 35]
#  [26 27 28 29 30]
#  [21 22 23 24 25]
#  [16 17 18 19 20]
#  [11 12 13 14 15]]

print(x[:, ::-1])
# [[15 14 13 12 11]
#  [20 19 18 17 16]
#  [25 24 23 22 21]
#  [30 29 28 27 26]
#  [35 34 33 32 31]]
```

通过对每个以逗号分隔的维度执行单独的切片，我们可以对多维数组进行切片。对于二维数组，我们的第一片定义了**行**的切片，第二片定义了**列**的切片。<br/>

【例】

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
print(x)
# [[11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]
#  [26 27 28 29 30]
#  [31 32 33 34 35]]

x[0::2, 1::3] = 0
print(x)
# [[11  0 13 14  0]
#  [16 17 18 19 20]
#  [21  0 23 24  0]
#  [26 27 28 29 30]
#  [31  0 33 34  0]]
```

#### dots 索引

Numpy 允许使用 `...` 表示足够多的冒号来构建完整的索引列表。<br/>

比如，如果 `x` 是五维数组：

- `x[1, 2, ...]` 等于 `x[1, 2, :, :, :]`
- `x[..., 3]` 等于 `x[:, :, :, :, 3]`
- `x[4, ..., 5, :]` 等于 `x[4, :, :, 5, :]`

【例】

```python
import numpy as np

x = np.random.randint(1, 100, [2, 2, 3])
print(x)
# [[[ 5 64 75]
#   [57 27 31]]
# 
#  [[68 85  3]
#   [93 26 25]]]

print(x[1, ...])
# [[68 85  3]
#  [93 26 25]]

print(x[..., 2])
# [[75 31]
#  [ 3 25]]
```

#### 整数数组索引

【例】方括号内传入多个索引值，可以同时选择多个元素。

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
r = [0, 1, 2]
print(x[r])
# [1 2 3]

r = [0, 1, -1]
print(x[r])
# [1 2 8]

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])

r = [0, 1, 2]
print(x[r])
# [[11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]]

r = [0, 1, -1]
print(x[r])

# [[11 12 13 14 15]
#  [16 17 18 19 20]
#  [31 32 33 34 35]]

r = [0, 1, 2]
c = [2, 3, 4]
y = x[r, c]
print(y)
# [13 19 25]
```

【例】

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
r = np.array([[0, 1], [3, 4]])
print(x[r])
# [[1 2]
#  [4 5]]

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])

r = np.array([[0, 1], [3, 4]])
print(x[r])
# [[[11 12 13 14 15]
#   [16 17 18 19 20]]
#
#  [[26 27 28 29 30]
#   [31 32 33 34 35]]]

# 获取了 5X5 数组中的四个角的元素。
# 行索引是 [0,0] 和 [4,4]，而列索引是 [0,4] 和 [0,4]。
r = np.array([[0, 0], [4, 4]])
c = np.array([[0, 4], [0, 4]])
y = x[r, c]
print(y)
# [[11 15]
#  [31 35]]
```

【例】可以借助切片 `:` 与整数数组的组合。

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])

y = x[0:3, [1, 2, 2]]
print(y)
# [[12 13 13]
#  [17 18 18]
#  [22 23 23]]
```

- `numpy.take(a, indices, axis=None, out=None, mode='raise')` Take elements from an array along an axis.

【例】

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
r = [0, 1, 2]
print(np.take(x, r))
# [1 2 3]

r = [0, 1, -1]
print(np.take(x, r))
# [1 2 8]

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])

r = [0, 1, 2]
print(np.take(x, r, axis=0))
# [[11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]]

r = [0, 1, -1]
print(np.take(x, r, axis=0))
# [[11 12 13 14 15]
#  [16 17 18 19 20]
#  [31 32 33 34 35]]

r = [0, 1, 2]
c = [2, 3, 4]
y = np.take(x, [r, c])
print(y)
# [[11 12 13]
#  [13 14 15]]
```

【注】使用切片索引到 numpy 数组时，生成的数组视图将始终是原始数组的子数组，但是整数数组索引产生的不是其子数组，而是形成新的数组。

- 切片索引

```python
import numpy as np

a = np.array([[1, 2], [3, 4], [5, 6]])
b = a[0:1, 0:1]
b[0, 0] = 2
print(a[0, 0] == b)
# [[True]]
```

- 整数数组索引

```python
import numpy as np

a = np.array([[1, 2], [3, 4], [5, 6]])
b = a[0, 0]
b = 2
print(a[0, 0] == b)
# False
```

#### 布尔索引

可以通过一个布尔数组来索引目标数组。

【例】

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = x > 5
print(y)
# [False False False False False  True  True  True]
print(x[x > 5])
# [6 7 8]

x = np.array([np.nan, 1, 2, np.nan, 3, 4, 5])
y = np.logical_not(np.isnan(x))
print(x[y])
# [1. 2. 3. 4. 5.]

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = x > 25
print(y)
# [[False False False False False]
#  [False False False False False]
#  [False False False False False]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]]
print(x[x > 25])
# [26 27 28 29 30 31 32 33 34 35]
```

【例】

```python
import numpy as np

import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 50)
y = np.sin(x)
print(len(x))  # 50
plt.plot(x, y)

mask = y >= 0
print(len(x[mask]))  # 25
print(mask)
'''
[ True  True  True  True  True  True  True  True  True  True  True  True
  True  True  True  True  True  True  True  True  True  True  True  True
  True False False False False False False False False False False False
 False False False False False False False False False False False False
 False False]
'''
plt.plot(x[mask], y[mask], 'bo')

mask = np.logical_and(y >= 0, x <= np.pi / 2)
print(mask)
'''
[ True  True  True  True  True  True  True  True  True  True  True  True
  True False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False]
'''

plt.plot(x[mask], y[mask], 'go')
plt.show()
```

![例图](https://gitee.com/Miraclezjy/utoolspic/raw/master/pic2-2022-1-1615:53:35.png)

我们利用这些条件来选择图上的不同点。蓝色点（在图中还包括绿点，但绿点掩盖了蓝色点）显示值大于 0 的所有点，绿色点表示值大于 0 且小于 0.5 $\pi$ 的所有点。

------

#### 数组迭代

除了 `for` 循环，Numpy 还提供另外一种更为优雅的遍历方法。

- `apply_along_axis(func1d, axis, arr)` Apply a function to 1-D slices along the given axis.

【例】

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])

y = np.apply_along_axis(np.sum, 0, x)
print(y)  # [105 110 115 120 125]
y = np.apply_along_axis(np.sum, 1, x)
print(y)  # [ 65  90 115 140 165]

y = np.apply_along_axis(np.mean, 0, x)
print(y)  # [21. 22. 23. 24. 25.]
y = np.apply_along_axis(np.mean, 1, x)
print(y)  # [13. 18. 23. 28. 33.]


def my_func(x):
    return (x[0] + x[-1]) * 0.5


y = np.apply_along_axis(my_func, 0, x)
print(y)  # [21. 22. 23. 24. 25.]
y = np.apply_along_axis(my_func, 1, x)
print(y)  # [13. 18. 23. 28. 33.]
```

- `axis=0`：列
- `axis=1`：行

# 第三章 数组的操作、变形

## 数组操作

### 更改形状

在对数组进行操作时，为了满足格式和计算的要求通常会改变其形状。

- `numpy.ndarray.shape` 表示数组的维度，返回一个元组，这个元组的长度就是维度的数目，即 `ndim` 属性（秩）。

【例】通过修改 `shape` 属性来改变数组的形状。

```python
import numpy as np

x = np.array([1, 2, 9, 4, 5, 6, 7, 8])
print(x.shape)  # (8,)
x.shape = [2, 4]
print(x)
# [[1 2 9 4]
#  [5 6 7 8]]
```

【注】`shape` 属性设置不当会报错，比如上例中若添加语句 `x.shape = [2, 3]`，会报错 `ValueError: cannot reshape array of size 8 into shape (2,3)`。

- `numpy.ndarray.flat` 将数组转换为一维的迭代器，可以用 `for` 访问数组每一个元素。

【例】

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = x.flat
print(y)
# <numpy.flatiter object at 0x0000020F9BA10C60>
for i in y:
    print(i, end=' ')
# 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35

y[3] = 0
print(end='\n')
print(x)
# [[11 12 13  0 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]
#  [26 27 28 29 30]
#  [31 32 33 34 35]]
```

- `numpy.ndarray.flatten([order='C'])` 将数组的**副本**转换为一维数组，并返回。
  - `order`：`‘C’`——按行，`‘F’`——按列，`‘A’`——按原顺序，`‘K’`——元素在内存中的出现顺序
  - `order`：`{'C' / 'F', 'A', 'K'}` 可选使用此索引顺序读取 a 的元素。`‘C’` 意味着以行大（注：行序优先）的 `C` 风格顺序对元素进行索引，最后一个轴索引会更改 `F` 表示以列大的 `Fortran` 样式顺序索引元素，其中第一个索引变化最快，最后一个索引变化最快。请注意，`'C'` 和 `'F'` 选项不考虑基础数组的内存布局，仅引用轴索引的顺序。`‘A’` 表示如果 a 为 `Fortran`，则以类似 `Fortran` 的索引顺序读取元素在内存中连续，否则类似 `C` 的顺序。`‘K’` 表示按照步序在内存中的顺序读取元素，但步幅为负时反转数据除外。默认情况下，使用 `Cindex` 顺序。

【例】`flatten()` 函数返回的是**拷贝**。

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = x.flatten()
print(y)
# [11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35]

y[3] = 0
print(x)
# [[11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]
#  [26 27 28 29 30]
#  [31 32 33 34 35]]

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = x.flatten(order='F')
print(y)
# [11 16 21 26 31 12 17 22 27 32 13 18 23 28 33 14 19 24 29 34 15 20 25 30 35]

y[3] = 0
print(x)
# [[11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]
#  [26 27 28 29 30]
#  [31 32 33 34 35]]
```

- `numpy.ravel(a, order='C')` Return a contiguous flattened array.

【例】`ravel()` 返回的是视图。

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.ravel(x)  # 等价于 y = np.ravel(x, order='C')
print(y)
# [11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35]

y[3] = 0
print(x)
# [[11 12 13  0 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]
#  [26 27 28 29 30]
#  [31 32 33 34 35]]
```

【例】`order='F'` 就是拷贝。

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])

y = np.ravel(x, order='F')
print(y)
# [11 16 21 26 31 12 17 22 27 32 13 18 23 28 33 14 19 24 29 34 15 20 25 30 35]

y[3] = 0
print(x)
# [[11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]
#  [26 27 28 29 30]
#  [31 32 33 34 35]]
```

- `numpy.reshape(a, newshape[, order='C'])` 在不更改数据的情况下为数组赋予新的形状。

【例】`reshape()` 函数当参数 `newshape = [rows, -1]` 时，将根据行数自动确定列数。

```python
import numpy as np

x = np.arange(12)
y = np.reshape(x, [3, 4])
print(y.dtype)  # int32
print(y)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

y = np.reshape(x, [3, -1])
print(y)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

y = np.reshape(x, [-1, 3])
print(y)
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]

y[0, 1] = 10
print(x)
# [ 0 10  2  3  4  5  6  7  8  9 10 11]（改变x去reshape后y中的值，x对应元素也改变）
```

【例】`reshape()`  函数当参数 `newshape = -1`  时，表示将数组降为一维。

```python
import numpy as np

x = np.random.randint(12, size=[2, 2, 3])
print(x)
# [[[ 3  8  1]
#   [10  3  9]]
# 
#  [[11  1  5]
#   [ 6 11 10]]]

y = np.reshape(x, -1)
print(y)
# [ 3  8  1 10  3  9 11  1  5  6 11 10]
```

### 数组转置

-  `numpy.transpose(a, axes=None)` Permute the dimensions of an array.
- `numpy.ndarray.T` Same as self.transpose(), except that self is returned if self.ndim < 2.

【例】

```python
import numpy as np

x = np.random.rand(5, 5) * 10
x = np.around(x, 2)
print(x)
# [[6.74 8.46 6.74 5.45 1.25]
#  [3.54 3.49 8.62 1.94 9.92]
#  [5.03 7.22 1.6  8.7  0.43]
#  [7.5  7.31 5.69 9.67 7.65]
#  [1.8  9.52 2.78 5.87 4.14]]
y = x.T
print(y)
# [[6.74 3.54 5.03 7.5  1.8 ]
#  [8.46 3.49 7.22 7.31 9.52]
#  [6.74 8.62 1.6  5.69 2.78]
#  [5.45 1.94 8.7  9.67 5.87]
#  [1.25 9.92 0.43 7.65 4.14]]
y = np.transpose(x)
print(y)
# [[6.74 3.54 5.03 7.5  1.8 ]
#  [8.46 3.49 7.22 7.31 9.52]
#  [6.74 8.62 1.6  5.69 2.78]
#  [5.45 1.94 8.7  9.67 5.87]
#  [1.25 9.92 0.43 7.65 4.14]]
```

【注】以上两个方法返回的都是**视图**。

### 更改维度

当创建一个数组之后，还可以给它增加一个维度，这在矩阵计算中经常会用到。

- `numpy.newaxis = None` None 的别名，对索引数组很有用。

【例】很多工具包在进行计算时都会先判断输入数据的维度是否满足要求，如果输入数据达不到指定的维度时，可以使用 `newaxis` 来增加一个维度。

```python
import numpy as np

x = np.array([1, 2, 9, 4, 5, 6, 7, 8])
print(x.shape)  # (8,)
print(x)  # [1 2 9 4 5 6 7 8]

y = x[np.newaxis, :]
print(y.shape)  # (1, 8)
print(y)  # [[1 2 9 4 5 6 7 8]]

y = x[:, np.newaxis]
print(y.shape)  # (8, 1)
print(y)
# [[1]
#  [2]
#  [9]
#  [4]
#  [5]
#  [6]
#  [7]
#  [8]]
```

- `numpy.squeeze(a, axis=None)` 从数组的形状中删除单维度条目，即把 shape 中为 1 的维度去掉。
  - a 表示输入的数组
  - axis 用于指定需要删除的维度，但是指定的维度必须为单维度，否则将会报错。

在机器学习和深度学习中，通常算法的结果是可以表示向量的数组（即包含两对或以上的方括号形式 [[]]），如果直接利用这个数组进行画图可能显示界面为空。我们可以利用 `squeeze()` 函数将表示向量的数组转换为秩为 1 的数组，这样利用 `matplotlib` 库函数画图时，就可以正常地显示结果了。

【例】删除单维度。

```python
import numpy as np

x = np.arange(10)
print(x.shape)  # (10,)
x = x[np.newaxis, :]
print(x.shape)  # (1, 10)
y = np.squeeze(x)
print(y.shape)  # (10,)
```

【例】

```python
import numpy as np

x = np.array([[[0], [1], [2]]])
print(x.shape)  # (1, 3, 1)
print(x)
# [[[0]
#   [1]
#   [2]]]

y = np.squeeze(x)
print(y.shape)  # (3,)
print(y)  # [0 1 2]

y = np.squeeze(x, axis=0)
print(y.shape)  # (3, 1)
print(y)
# [[0]
#  [1]
#  [2]]

y = np.squeeze(x, axis=2)
print(y.shape)  # (1, 3)
print(y)  # [[0 1 2]]

y = np.squeeze(x, axis=1)
# ValueError: cannot select an axis to squeeze out which has size not equal to one
```

【例】修正维度后作图。

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1, 4, 9, 16, 25]])
x = np.squeeze(x)
print(x.shape)  # (5, )
plt.plot(x)
plt.show()
```

![例图](https://gitee.com/Miraclezjy/utoolspic/raw/master/pic1-2022-1-1612:14:14.png)

### 数组组合

如果要将两份数据合在一起，就需要拼接操作。

- `numpy.concatenate((a1, a2, ...), axis=0, out=None)` Join a sequence of arrays along an existing axis.

【例】连接沿现有轴的数组序列（原来 x, y 都是一维的，拼接后的结果也是一维的）。

```python
import numpy as np

x = np.array([1, 2, 3])
y = np.array([7, 8, 9])
z = np.concatenate([x, y])
print(z)
# [1 2 3 7 8 9]

z = np.concatenate([x, y], axis=0)
print(z)
# [1 2 3 7 8 9]
```

【例】原来 x, y 都是二维的，拼接后的结果也是二维的。

```python
import numpy as np

x = np.array([1, 2, 3]).reshape(1, 3)
y = np.array([7, 8, 9]).reshape(1, 3)
z = np.concatenate([x, y])
print(z)
# [[ 1  2  3]
#  [ 7  8  9]]
z = np.concatenate([x, y], axis=0)
print(z)
# [[ 1  2  3]
#  [ 7  8  9]]
z = np.concatenate([x, y], axis=1)
print(z)
# [[ 1  2  3  7  8  9]]
```

【例】x, y 在原来的维度上拼接。

```python
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[7, 8, 9], [10, 11, 12]])
z = np.concatenate([x, y])
print(z)
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]
z = np.concatenate([x, y], axis=0)
print(z)
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]
z = np.concatenate([x, y], axis=1)
print(z)
# [[ 1  2  3  7  8  9]
#  [ 4  5  6 10 11 12]]
```

- `numpy.stack(arrays, axis=0, out=None)` Join a sequence of arrays along a new axis.

【例】沿着新的轴加入一系列数组（stack 为增加维度的拼接）

```python
import numpy as np

x = np.array([1, 2, 3])
y = np.array([7, 8, 9])
z = np.stack([x, y])
print(z.shape)  # (2, 3)
print(z)
# [[1 2 3]
#  [7 8 9]]

z = np.stack([x, y], axis=1)
print(z.shape)  # (3, 2)
print(z)
# [[1 7]
#  [2 8]
#  [3 9]]
```

【例】

```python
import numpy as np

x = np.array([1, 2, 3]).reshape(1, 3)
y = np.array([7, 8, 9]).reshape(1, 3)
z = np.stack([x, y])
print(z.shape)  # (2, 1, 3)
print(z)
# [[[1 2 3]]
#  [[7 8 9]]]

z = np.stack([x, y], axis=1)
print(z.shape)  # (1, 2, 3)
print(z)
# [[[1 2 3]
#   [7 8 9]]]

z = np.stack([x, y], axis=2)
print(z.shape)  # (1, 3, 2)
print(z)
# [[[1 7]
#   [2 8]
#   [3 9]]]
```

【例】

```python
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[7, 8, 9], [10, 11, 12]])
z = np.stack([x, y])
print(z.shape)  # (2, 2, 3)
print(z)
# [[[ 1  2  3]
#   [ 4  5  6]]
#
#  [[ 7  8  9]
#   [10 11 12]]]

z = np.stack([x, y], axis=1)
print(z.shape)  # (2, 2, 3)
print(z)
# [[[ 1  2  3]
#   [ 7  8  9]]
#
#  [[ 4  5  6]
#   [10 11 12]]]

z = np.stack([x, y], axis=2)
print(z.shape)  # (2, 3, 2)
print(z)
# [[[ 1  7]
#   [ 2  8]
#   [ 3  9]]
#
#  [[ 4 10]
#   [ 5 11]
#   [ 6 12]]]
```

- `numpy.vstack(tup)` Stack arrays in sequence vertically (row wise).
- `numpy.hstack(tup)` Stack arrays in sequence horizontally (column wise).

【例】一维的情况

```python
import numpy as np

x = np.array([1, 2, 3])
y = np.array([7, 8, 9])
z = np.vstack((x, y))
print(z.shape)  # (2, 3)
print(z)
# [[1 2 3]
#  [7 8 9]]

z = np.stack([x, y])
print(z.shape)  # (2, 3)
print(z)
# [[1 2 3]
#  [7 8 9]]

z = np.hstack((x, y))
print(z.shape)  # (6,)
print(z)
# [1  2  3  7  8  9]

z = np.concatenate((x, y))
print(z.shape)  # (6,)
print(z)  # [1 2 3 7 8 9]
```

【例】二维的情况

```python
import numpy as np

x = np.array([1, 2, 3]).reshape(1, 3)
y = np.array([7, 8, 9]).reshape(1, 3)
z = np.vstack((x, y))
print(z.shape)  # (2, 3)
print(z)
# [[1 2 3]
#  [7 8 9]]

z = np.concatenate((x, y), axis=0)
print(z.shape)  # (2, 3)
print(z)
# [[1 2 3]
#  [7 8 9]]

z = np.hstack((x, y))
print(z.shape)  # (1, 6)
print(z)
# [[1 2 3 7 8 9]]

z = np.concatenate((x, y), axis=1)
print(z.shape)  # (1, 6)
print(z)
# [[1 2 3 7 8 9]]
```

【例】二维的情况

```python
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[7, 8, 9], [10, 11, 12]])
z = np.vstack((x, y))
print(z.shape)  # (4, 3)
print(z)
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]

z = np.concatenate((x, y), axis=0)
print(z.shape)  # (4, 3)
print(z)
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]

z = np.hstack((x, y))
print(z.shape)  # (2, 6)
print(z)
# [[ 1  2  3  7  8  9]
#  [ 4  5  6 10 11 12]]

z = np.concatenate((x, y), axis=1)
print(z.shape)  # (2, 6)
print(z)
# [[ 1  2  3  7  8  9]
#  [ 4  5  6 10 11 12]]
```

`hstack()`、`vstack()` 分别表示水平和竖直的拼接方式。在数据维度等于 1 时，比较特殊；而当维度大于或等于 2 时，它们的作用相当与 `concatenate`，用于在已有轴上进行操作。

【例】

```python
import numpy as np

a = np.hstack([np.array([1, 2, 3, 4]), 5])
print(a)  # [1 2 3 4 5]

a = np.concatenate([np.array([1, 2, 3, 4]), 5])
print(a)
# all the input arrays must have same number of dimensions, but the array at index 0 has 1 dimension(s) and the array at index 1 has 0 dimension(s)
```

### 数组拆分

- `numpy.split(ary, indices_or_sections, axis=0)` Split an array into multiple sub-arrays as views into ary.

【例 】拆分数组

```python
import numpy as np

x = np.array([[11, 12, 13, 14],
              [16, 17, 18, 19],
              [21, 22, 23, 24]])
y = np.split(x, [1, 3])
print(y)
# [array([[11, 12, 13, 14]]), array([[16, 17, 18, 19],
#        [21, 22, 23, 24]]), array([], shape=(0, 4), dtype=int32)]

y = np.split(x, [1, 3], axis=1)
print(y)
# [array([[11],
#        [16],
#        [21]]), array([[12, 13],
#        [17, 18],
#        [22, 23]]), array([[14],
#        [19],
#        [24]])]
```

- `numpy.vsplit(ary, indices_or_sections)` Split an array into multiple sub-arrays vertically (row wise).

【例】垂直切分是把数组按照高度切分

```python
import numpy as np

x = np.array([[11, 12, 13, 14],
              [16, 17, 18, 19],
              [21, 22, 23, 24]])
y = np.vsplit(x, 3)
print(y)
# [array([[11, 12, 13, 14]]), array([[16, 17, 18, 19]]), array([[21, 22, 23, 24]])]

y = np.split(x, 3)
print(y)
# [array([[11, 12, 13, 14]]), array([[16, 17, 18, 19]]), array([[21, 22, 23, 24]])]


y = np.vsplit(x, [1])
print(y)
# [array([[11, 12, 13, 14]]), array([[16, 17, 18, 19],
#        [21, 22, 23, 24]])]

y = np.split(x, [1])
print(y)
# [array([[11, 12, 13, 14]]), array([[16, 17, 18, 19],
#        [21, 22, 23, 24]])]


y = np.vsplit(x, [1, 3])
print(y)
# [array([[11, 12, 13, 14]]), array([[16, 17, 18, 19],
#        [21, 22, 23, 24]]), array([], shape=(0, 4), dtype=int32)]
y = np.split(x, [1, 3], axis=0)
print(y)
# [array([[11, 12, 13, 14]]), array([[16, 17, 18, 19],
#        [21, 22, 23, 24]]), array([], shape=(0, 4), dtype=int32)]
```

- `numpy.hsplit(ary, indices_or_sections)` Split an array into multiple sub-arrays horizontally (column wise).

【例】水平切分是把数组按照宽度切分

```python
import numpy as np

x = np.array([[11, 12, 13, 14],
              [16, 17, 18, 19],
              [21, 22, 23, 24]])
y = np.hsplit(x, 2)
print(y)
# [array([[11, 12],
#        [16, 17],
#        [21, 22]]), array([[13, 14],
#        [18, 19],
#        [23, 24]])]

y = np.split(x, 2, axis=1)
print(y)
# [array([[11, 12],
#        [16, 17],
#        [21, 22]]), array([[13, 14],
#        [18, 19],
#        [23, 24]])]

y = np.hsplit(x, [3])
print(y)
# [array([[11, 12, 13],
#        [16, 17, 18],
#        [21, 22, 23]]), array([[14],
#        [19],
#        [24]])]

y = np.split(x, [3], axis=1)
print(y)
# [array([[11, 12, 13],
#        [16, 17, 18],
#        [21, 22, 23]]), array([[14],
#        [19],
#        [24]])]

y = np.hsplit(x, [1, 3])
print(y)
# [array([[11],
#        [16],
#        [21]]), array([[12, 13],
#        [17, 18],
#        [22, 23]]), array([[14],
#        [19],
#        [24]])]

y = np.split(x, [1, 3], axis=1)
print(y)
# [array([[11],
#        [16],
#        [21]]), array([[12, 13],
#        [17, 18],
#        [22, 23]]), array([[14],
#        [19],
#        [24]])]
```

### 数组平铺

- `numpy.tile(A, reps)` Construct an array by repeating A the number of times given by reps.

`tile` 是瓷砖的意思，顾名思义，这个函数就是把数组像瓷砖一样铺展开来。

【例】将原矩阵横向、纵向复制。

```python
import numpy as np

x = np.array([[1, 2], [3, 4]])
print(x)
# [[1 2]
#  [3 4]]

y = np.tile(x, (1, 3))
print(y)
# [[1 2 1 2 1 2]
#  [3 4 3 4 3 4]]

y = np.tile(x, (3, 1))
print(y)
# [[1 2]
#  [3 4]
#  [1 2]
#  [3 4]
#  [1 2]
#  [3 4]]

y = np.tile(x, (3, 3))
print(y)
# [[1 2 1 2 1 2]
#  [3 4 3 4 3 4]
#  [1 2 1 2 1 2]
#  [3 4 3 4 3 4]
#  [1 2 1 2 1 2]
#  [3 4 3 4 3 4]]
```

- `numpy.repeat(a, repeats, axis=None)` Repeat elements of an array.
  - `axis=0`：沿着 y 轴复制，实际上增加了行数；
  - `axis=1`：沿着 x 轴复制，实际上增加了列数；
  - `repeats`：可以为一个数，也可以为一个矩阵；
  - `axis=None`：此时会 flatten 当前矩阵，实际上就是变成了一个行向量。

【例】重复数组的元素。

```python
import numpy as np

x = np.repeat(3, 4)
print(x)  # [3 3 3 3]

x = np.array([[1, 2], [3, 4]])
y = np.repeat(x, 2)
print(y)
# [1 1 2 2 3 3 4 4]

y = np.repeat(x, 2, axis=0)
print(y)
# [[1 2]
#  [1 2]
#  [3 4]
#  [3 4]]

y = np.repeat(x, 2, axis=1)
print(y)
# [[1 1 2 2]
#  [3 3 4 4]]

y = np.repeat(x, [2, 3], axis=0)
print(y)
# [[1 2]
#  [1 2]
#  [3 4]
#  [3 4]
#  [3 4]]

y = np.repeat(x, [2, 3], axis=1)
print(y)
# [[1 1 2 2 2]
#  [3 3 4 4 4]]
```

------

## 添加和删除元素

- `numpy.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None)` Find the unique elements of an array.
  - `return_index`：the indices of the input array that give the unique values.
  - `return_inverse`：the indices of the unique array that reconstruct the input array.
  - `return_counts`：the number of times each unique value comes up in the input array.

【例】查找数组的唯一元素

```python
import numpy as np

a = np.array([1, 1, 2, 3, 3, 4, 4])
b = np.unique(a, return_counts=True)
print(b[0][list(b[1]).index(1)])  # 2
```

## 【练习】

### 1.

- stem

将 arr 转换为 2 行的二维数组

```python
arr = np.arange(10)
```

- code

```python
import numpy as np

arr = np.arange(10)

# 方法一
x = np.reshape(arr, newshape=[2, 5])
print(x)
# [[0 1 2 3 4]
#  [5 6 7 8 9]]

# 方法二
x = np.reshape(arr, newshape=[2, -1])
print(x)
# [[0 1 2 3 4]
#  [5 6 7 8 9]]
```

### 2.

- stem

垂直堆叠数组 a 和数组 b

```python
a = np.arange(10).reshape([2, -1])
b = np.repeat(1, 10).reshape([2, -1])
```

- code

```python
import numpy as np

a = np.arange(10).reshape([2, -1])
b = np.repeat(1, 10).reshape([2, -1])

print(a)
# [[0 1 2 3 4]
#  [5 6 7 8 9]]
print(b)
# [[1 1 1 1 1]
#  [1 1 1 1 1]]

# 方法一
print(np.concatenate([a, b], axis=0))
# [[0 1 2 3 4]
#  [5 6 7 8 9]
#  [1 1 1 1 1]
#  [1 1 1 1 1]]

# 方法二
print(np.vstack([a, b]))
# [[0 1 2 3 4]
#  [5 6 7 8 9]
#  [1 1 1 1 1]
#  [1 1 1 1 1]]
```

### 3.

- stem

水平堆叠数组 a 和数组 b

```python
a = np.arange(10).reshape([2, -1])
b = np.repeat(1, 10).reshape([2, -1])
```

- code

```python
import numpy as np

a = np.arange(10).reshape([2, -1])
b = np.repeat(1, 10).reshape([2, -1])

print(a)
# [[0 1 2 3 4]
#  [5 6 7 8 9]]
print(b)
# [[1 1 1 1 1]
#  [1 1 1 1 1]]

# 方法一
print(np.concatenate([a, b], axis=1))
# [[0 1 2 3 4 1 1 1 1 1]
#  [5 6 7 8 9 1 1 1 1 1]]

# 方法二
print(np.hstack([a, b]))
# [[0 1 2 3 4 1 1 1 1 1]
#  [5 6 7 8 9 1 1 1 1 1]]
```

### 4.

- stem

将 arr 的 2 维数组按列输出

```python
arr = np.array([[16, 17, 18, 19, 20],[11, 12, 13, 14, 15],[21, 22, 23, 24, 25],[31, 32, 33, 34, 35],[26, 27, 28, 29, 30]])
```

- code

```python
import numpy as np

arr =  np.array([[16, 17, 18, 19, 20],[11, 12, 13, 14, 15],[21, 22, 23, 24, 25],[31, 32, 33, 34, 35],[26, 27, 28, 29, 30]])
x = arr.flatten(order='F')
print(x)
# [16 11 21 31 26 17 12 22 32 27 18 13 23 33 28 19 14 24 34 29 20 15 25 35 30]
```

### 5.

- stem

给定两个随机数组 A 和 B，验证它们是否相等

```python
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)
```

- code

```python
import numpy as np

A = np.array([1, 2, 3])
B = np.array([1, 2, 3])

# Assuming identical shape of the arrays and a tolerance for the comparison of values
equal = np.allclose(A, B)
print(equal)

# Checking both the shape and the element values, no tolerance (values have to be exactly equal)
equal = np.array_equal(A, B)
print(equal)
```

### 6.

- stem

在给定的 numpy 数组中找到重复的条目（第二次出现以后），并将它们标记为 True。第一次出现应为 False。

```python
a = np.random.randint(0, 5, 10)
```

- code

```python
import numpy as np

np.random.seed(100)
a = np.random.randint(0, 5, 10)
print(a)
# [0 0 3 0 2 4 2 2 2 2]
b = np.full(10, True)
vals, counts = np.unique(a, return_index=True)
b[counts] = False
print(b)
# [False  True False  True False False  True  True  True  True]
```

### 7.

- stem

建立一个随机数在 1-10 之间的 3 行 2 列的数组，并将其转换成 2 行 3 列的数组。

- code

```python
import numpy as np

# 建立 3 行 2 列数组
x = np.random.randint(10, size=[3, 2])
print(x)

# 转换为 2 行 3 列数组
y = np.reshape(x, [2,3])
print(y)
```

# 第四章 数学函数及逻辑函数

## 一、逻辑函数

### 真值测试

#### numpy.all

- `numpy.all(a, axis=None, out=None, keepdims=np._NoValue)`：Test whether all array elements along a given axis evaluate to True.

#### numpy.any

- `numpy.any(a, axis=None, out=None, keepdims=np._NoValue)`：Test whether any array element along a given axis evaluates to True.

【例】

```python
import numpy as np

a = np.array([0, 4, 5])
b = np.copy(a)
print(np.all(a == b))  # True
print(np.any(a == b))  # True

b[0] = 1
print(np.all(a == b))  # False
print(np.any(a == b))  # True

print(np.all([1.0, np.nan]))  # True
print(np.any([1.0, np.nan]))  # True

a = np.eye(3)
print(np.all(a, axis=0))  # [False False False]
print(np.any(a, axis=0))  # [ True  True  True]
```

------

### 数组内容

#### numpy.isnan

- `numpy.isnan(x, *args, **kwargs)`：Test element-wise for NaN and return result as a boolean array.

【例】

```python
import numpy as np

a = np.array([1, 2, np.nan])
print(np.isnan(a))
# [False False  True]
```

### 逻辑运算

#### numpy.logical_not

- `numpy.logical_not(x, *args, **kwargs)`：Compute the truth value of NOT x, element-wise.
- `numpy.logical_and(x1, x2, *args, **kwargs)`：Compute the truth value of x1 AND x2, element-wise.
- `numpy.logical_or(x1, x2, *args, **kwargs)`：Compute the truth value of x1 OR x2, element-wise.
- `numpy.logical_xor(x1, x2, *args, **kwargs)`：Compute the truth value of x1 XOR x2, element-wise.

【例】计算非 x 元素的真值。

```python
import numpy as np

print(np.logical_not(3))
# False
print(np.logical_not([True, False, 0, 1]))
# [False  True  True False]

x = np.arange(5)
print(np.logical_not(x < 3))
# [False False False  True  True]
```

【例】计算 x1 AND x2 元素的真值。

```python
print(np.logical_and(True, False))  
# False
print(np.logical_and([True, False], [True, False]))
# [ True False]
print(np.logical_and(x > 1, x < 4))
# [False False  True  True False]
```

【例】逐元素计算 x1 OR x2 的值。

```python
print(np.logical_or(True, False))
# True
print(np.logical_or([True, False], [False, False]))
# [ True False]
print(np.logical_or(x < 1, x > 3))
# [ True False False False  True]
```

【例】计算 x1 XOR x2 的真值，按元素计算。

```python
print(np.logical_xor(True, False))
# True
print(np.logical_xor([True, True, False, False], [True, False, True, False]))
# [False  True  True False]
print(np.logical_xor(x < 1, x > 3))
# [ True False False False  True]
print(np.logical_xor(0, np.eye(2)))
# [[ True False]
#  [False  True]]
```

### 对照

#### numpy.greater

- `numpy.greater(x1, x2, *args, **kwargs)`：Return the truth value of (x1 > x2), element-wise.

#### numpy.greater_equal

- `numpy.greater_equal(x1, x2, *args, **kwargs)`：Return the truth value of (x1 >= x2), element-wise.

#### numpy.equal

#### numpy.not_equal

#### numpy.less

#### numpy.less_equal

【例】numpy 对以上对照函数进行了运算符的重载。

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])

y = x > 2
print(y)
print(np.greater(x, 2))
# [False False  True  True  True  True  True  True]

y = x >= 2
print(y)
print(np.greater_equal(x, 2))
# [False  True  True  True  True  True  True  True]

y = x == 2
print(y)
print(np.equal(x, 2))
# [False  True False False False False False False]

y = x != 2
print(y)
print(np.not_equal(x, 2))
# [ True False  True  True  True  True  True  True]

y = x < 2
print(y)
print(np.less(x, 2))
# [ True False False False False False False False]

y = x <= 2
print(y)
print(np.less_equal(x, 2))
# [ True  True False False False False False False]
```

【例】

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = x > 20
print(y)
print(np.greater(x, 20))
# [[False False False False False]
#  [False False False False False]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]]

y = x >= 20
print(y)
print(np.greater_equal(x, 20))
# [[False False False False False]
#  [False False False False  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]]

y = x == 20
print(y)
print(np.equal(x, 20))
# [[False False False False False]
#  [False False False False  True]
#  [False False False False False]
#  [False False False False False]
#  [False False False False False]]

y = x != 20
print(y)
print(np.not_equal(x, 20))
# [[ True  True  True  True  True]
#  [ True  True  True  True False]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]]


y = x < 20
print(y)
print(np.less(x, 20))
# [[ True  True  True  True  True]
#  [ True  True  True  True False]
#  [False False False False False]
#  [False False False False False]
#  [False False False False False]]

y = x <= 20
print(y)
print(np.less_equal(x, 20))
# [[ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [False False False False False]
#  [False False False False False]
#  [False False False False False]]
```

【例】

```python
import numpy as np

np.random.seed(20200611)
x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])

y = np.random.randint(10, 40, [5, 5])
print(y)
# [[32 28 31 33 37]
#  [23 37 37 30 29]
#  [32 24 10 33 15]
#  [27 17 10 36 16]
#  [25 32 23 39 34]]

z = x > y
print(z)
print(np.greater(x, y))
# [[False False False False False]
#  [False False False False False]
#  [False False  True False  True]
#  [False  True  True False  True]
#  [ True False  True False  True]]

z = x >= y
print(z)
print(np.greater_equal(x, y))
# [[False False False False False]
#  [False False False False False]
#  [False False  True False  True]
#  [False  True  True False  True]
#  [ True  True  True False  True]]

z = x == y
print(z)
print(np.equal(x, y))
# [[False False False False False]
#  [False False False False False]
#  [False False False False False]
#  [False False False False False]
#  [False  True False False False]]

z = x != y
print(z)
print(np.not_equal(x, y))
# [[ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True False  True  True  True]]

z = x < y
print(z)
print(np.less(x, y))
# [[ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True False  True False]
#  [ True False False  True False]
#  [False False False  True False]]

z = x <= y
print(z)
print(np.less_equal(x, y))
# [[ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True False  True False]
#  [ True False False  True False]
#  [False  True False  True False]]
```

【例】注意 numpy 的广播规则。

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])

np.random.seed(20200611)
y = np.random.randint(10, 50, 5)

print(y)
# [32 37 30 24 10]

z = x > y
print(z)
print(np.greater(x, y))
# [[False False False False  True]
#  [False False False False  True]
#  [False False False False  True]
#  [False False False  True  True]
#  [False False  True  True  True]]

z = x >= y
print(z)
print(np.greater_equal(x, y))
# [[False False False False  True]
#  [False False False False  True]
#  [False False False  True  True]
#  [False False False  True  True]
#  [False False  True  True  True]]

z = x == y
print(z)
print(np.equal(x, y))
# [[False False False False False]
#  [False False False False False]
#  [False False False  True False]
#  [False False False False False]
#  [False False False False False]]

z = x != y
print(z)
print(np.not_equal(x, y))
# [[ True  True  True  True  True]
#  [ True  True  True  True  True]
#  [ True  True  True False  True]
#  [ True  True  True  True  True]
#  [ True  True  True  True  True]]

z = x < y
print(z)
print(np.less(x, y))
# [[ True  True  True  True False]
#  [ True  True  True  True False]
#  [ True  True  True False False]
#  [ True  True  True False False]
#  [ True  True False False False]]

z = x <= y
print(z)
print(np.less_equal(x, y))
# [[ True  True  True  True False]
#  [ True  True  True  True False]
#  [ True  True  True  True False]
#  [ True  True  True False False]
#  [ True  True False False False]]
```

#### numpy.isclose

- `numpy.isclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False)`：Returns a boolean array where two arrays are element-wise equal within a tolerance.

#### numpy.allclose

- `numpy.allclose(a, b, rtol=1.3-5, atol=1.e-8, equal_nan=False)`：Returns True if two arrays are element-wise equal within a tolerance.

`numpy.allclose()` 等价于 `numpy.all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))`。

The tolerance values are positive, typically very small numbers. The relative difference (`rtol * abs(b)`) and the absolute difference `atol` are added together to compare the absolute difference between `a` and `b`.<br/>

判断是否为 True 的计算依据：<br/>

`np.absolute(a - b) <= (atol + rtol * absolute(b))`

- `atol`：float，绝对公差
- `rtol`：float，相对公差

NaNs are treated as equal if they are in the same place and if `equal_nan=True`. Infs are treated as equal if they are in the same place and of the same sign in both arrays.<br/>

【例】比较两个数组是否可以认为相等

```python
import numpy as np

x = np.isclose([1e10, 1e-7], [1.00001e10, 1e-8])
print(x)  # [ True False]

x = np.allclose([1e10, 1e-7], [1.00001e10, 1e-8])
print(x)  # False

x = np.isclose([1e10, 1e-8], [1.00001e10, 1e-9])
print(x)  # [ True  True]

x = np.allclose([1e10, 1e-8], [1.00001e10, 1e-9])
print(x)  # True

x = np.isclose([1e10, 1e-8], [1.0001e10, 1e-9])
print(x)  # [False  True]

x = np.allclose([1e10, 1e-8], [1.0001e10, 1e-9])
print(x)  # False

x = np.isclose([1.0, np.nan], [1.0, np.nan])
print(x)  # [ True False]

x = np.allclose([1.0, np.nan], [1.0, np.nan])
print(x)  # False

x = np.isclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
print(x)  # [ True  True]

x = np.allclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
print(x)  # True
```

## 二、数学函数

### 向量化和广播

向量化和广播这两个概念是 numpy 内部实现的基础。有了向量化，编写代码时无需使用显式循环。这些循环实际上不能省略，只不过是在内部实现，被代码中的其他结构代替。向量化的应用使得代码更简洁，可读性更强，也可以说使用了向量化方法的代码看上去更“Pythonic”。<br/>

广播（Broadcasting）机制描述了 numpy 如何在算数运算期间处理具有不同形状的数组，让较小的数组在较大的数组上“广播”，以便它们具有兼容的形状。并不是所有的维度都要彼此兼容才符合广播机制的要求，但它们必须满足一定的条件。<br/>

若两个数组的各维度兼容，也就是两个数组的每一维等长，或其中一个数组为一维，那么广播机制就适用。如果这两个条件不满足，numpy 就会抛出异常，说两个数组不兼容。<br/>

总结来说，广播的规则有三个：

- 如果两个数组的维度数 dim 不相同，那么小维度数组的形状将会在左边补 1；
- 如果 shape 维度不匹配，但是有维度是 1，那么可以拓展维度是 1 的维度匹配另一个数组；
- 如果 shape 维度不匹配，但是没有任何一个维度是 1，则匹配引发错误。

【例】二维数组加一维数组

```python
import numpy as np

x = np.arange(4)
y = np.ones((3, 4))
print(x.shape)  # (4,)
print(y.shape)  # (3, 4)

print((x + y).shape)  # (3, 4)
print(x + y)
# [[1. 2. 3. 4.]
#  [1. 2. 3. 4.]
#  [1. 2. 3. 4.]]
```

【例】两个数组都需要广播

```python
import numpy as np

x = np.arange(4).reshape(4, 1)
y = np.ones(5)

print(x.shape)  # (4, 1)
print(y.shape)  # (5,)

print((x + y).shape)  # (4, 5)
print(x + y)
# [[1. 1. 1. 1. 1.]
#  [2. 2. 2. 2. 2.]
#  [3. 3. 3. 3. 3.]
#  [4. 4. 4. 4. 4.]]

x = np.array([0.0, 10.0, 20.0, 30.0])
y = np.array([1.0, 2.0, 3.0])
z = x[:, np.newaxis] + y
print(z)
# [[ 1.  2.  3.]
#  [11. 12. 13.]
#  [21. 22. 23.]
#  [31. 32. 33.]]
```

【例】不匹配报错的例子

```python
import numpy as np

x = np.arange(4)
y = np.ones(5)

print(x.shape)  # (4,)
print(y.shape)  # (5,)

print(x + y)
# ValueError: operands could not be broadcast together with shapes (4,) (5,) 
```

------

### 算数运算

#### numpy.add

#### numpy.subtract

#### numpy.multiply

#### numpy.divide

#### numpy.floor_divide

#### numpy.power

在 numpy 中对以上函数进行了运算符的重载，且运算符为**元素级**。也就是说，它们只用于位置相同的元素之间，所得到的运算结果组成一个新的数组。<br/>

【例】注意 numpy 的广播规则

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = x + 1
print(y)
print(np.add(x, 1))
# [2 3 4 5 6 7 8 9]

y = x - 1
print(y)
print(np.subtract(x, 1))
# [0 1 2 3 4 5 6 7]

y = x * 2
print(y)
print(np.multiply(x, 2))
# [ 2  4  6  8 10 12 14 16]

y = x / 2
print(y)
print(np.divide(x, 2))
# [0.5 1.  1.5 2.  2.5 3.  3.5 4. ]

y = x // 2
print(y)
print(np.floor_divide(x, 2))
# [0 1 1 2 2 3 3 4]

y = x ** 2
print(y)
print(np.power(x, 2))
# [ 1  4  9 16 25 36 49 64]
```

【例】注意 numpy 的广播规则

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = x + 1
print(y)
print(np.add(x, 1))
# [[12 13 14 15 16]
#  [17 18 19 20 21]
#  [22 23 24 25 26]
#  [27 28 29 30 31]
#  [32 33 34 35 36]]

y = x - 1
print(y)
print(np.subtract(x, 1))
# [[10 11 12 13 14]
#  [15 16 17 18 19]
#  [20 21 22 23 24]
#  [25 26 27 28 29]
#  [30 31 32 33 34]]

y = x * 2
print(y)
print(np.multiply(x, 2))
# [[22 24 26 28 30]
#  [32 34 36 38 40]
#  [42 44 46 48 50]
#  [52 54 56 58 60]
#  [62 64 66 68 70]]

y = x / 2
print(y)
print(np.divide(x, 2))
# [[ 5.5  6.   6.5  7.   7.5]
#  [ 8.   8.5  9.   9.5 10. ]
#  [10.5 11.  11.5 12.  12.5]
#  [13.  13.5 14.  14.5 15. ]
#  [15.5 16.  16.5 17.  17.5]]

y = x // 2
print(y)
print(np.floor_divide(x, 2))
# [[ 5  6  6  7  7]
#  [ 8  8  9  9 10]
#  [10 11 11 12 12]
#  [13 13 14 14 15]
#  [15 16 16 17 17]]

y = x ** 2
print(y)
print(np.power(x, 2))
# [[ 121  144  169  196  225]
#  [ 256  289  324  361  400]
#  [ 441  484  529  576  625]
#  [ 676  729  784  841  900]
#  [ 961 1024 1089 1156 1225]]
```

【例】注意 numpy 的广播规则

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])

y = np.arange(1, 6)
print(y)
# [1 2 3 4 5]

z = x + y
print(z)
print(np.add(x, y))
# [[12 14 16 18 20]
#  [17 19 21 23 25]
#  [22 24 26 28 30]
#  [27 29 31 33 35]
#  [32 34 36 38 40]]

z = x - y
print(z)
print(np.subtract(x, y))
# [[10 10 10 10 10]
#  [15 15 15 15 15]
#  [20 20 20 20 20]
#  [25 25 25 25 25]
#  [30 30 30 30 30]]

z = x * y
print(z)
print(np.multiply(x, y))
# [[ 11  24  39  56  75]
#  [ 16  34  54  76 100]
#  [ 21  44  69  96 125]
#  [ 26  54  84 116 150]
#  [ 31  64  99 136 175]]

z = x / y
print(z)
print(np.divide(x, y))
# [[11.          6.          4.33333333  3.5         3.        ]
#  [16.          8.5         6.          4.75        4.        ]
#  [21.         11.          7.66666667  6.          5.        ]
#  [26.         13.5         9.33333333  7.25        6.        ]
#  [31.         16.         11.          8.5         7.        ]]

z = x // y
print(z)
print(np.floor_divide(x, y))
# [[11  6  4  3  3]
#  [16  8  6  4  4]
#  [21 11  7  6  5]
#  [26 13  9  7  6]
#  [31 16 11  8  7]]

z = x ** np.full([1, 5], 2)
print(z)
print(np.power(x, np.full([5, 5], 2)))
# [[ 121  144  169  196  225]
#  [ 256  289  324  361  400]
#  [ 441  484  529  576  625]
#  [ 676  729  784  841  900]
#  [ 961 1024 1089 1156 1225]]
```

【例】

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])

y = np.arange(1, 26).reshape([5, 5])
print(y)
# [[ 1  2  3  4  5]
#  [ 6  7  8  9 10]
#  [11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]]

z = x + y
print(z)
print(np.add(x, y))
# [[12 14 16 18 20]
#  [22 24 26 28 30]
#  [32 34 36 38 40]
#  [42 44 46 48 50]
#  [52 54 56 58 60]]

z = x - y
print(z)
print(np.subtract(x, y))
# [[10 10 10 10 10]
#  [10 10 10 10 10]
#  [10 10 10 10 10]
#  [10 10 10 10 10]
#  [10 10 10 10 10]]

z = x * y
print(z)
print(np.multiply(x, y))
# [[ 11  24  39  56  75]
#  [ 96 119 144 171 200]
#  [231 264 299 336 375]
#  [416 459 504 551 600]
#  [651 704 759 816 875]]

z = x / y
print(z)
print(np.divide(x, y))
# [[11.          6.          4.33333333  3.5         3.        ]
#  [ 2.66666667  2.42857143  2.25        2.11111111  2.        ]
#  [ 1.90909091  1.83333333  1.76923077  1.71428571  1.66666667]
#  [ 1.625       1.58823529  1.55555556  1.52631579  1.5       ]
#  [ 1.47619048  1.45454545  1.43478261  1.41666667  1.4       ]]

z = x // y
print(z)
print(np.floor_divide(x, y))
# [[11  6  4  3  3]
#  [ 2  2  2  2  2]
#  [ 1  1  1  1  1]
#  [ 1  1  1  1  1]
#  [ 1  1  1  1  1]]

z = x ** np.full([5, 5], 2)
print(z)
print(np.power(x, np.full([5, 5], 2)))
# [[ 121  144  169  196  225]
#  [ 256  289  324  361  400]
#  [ 441  484  529  576  625]
#  [ 676  729  784  841  900]
#  [ 961 1024 1089 1156 1225]]
```

#### numpy.sqrt

#### numpy.square

【例】

```python
import numpy as np

x = np.arange(1, 5)
print(x)  # [1 2 3 4]

y = np.sqrt(x)
print(y)
# [1.         1.41421356 1.73205081 2.        ]
print(np.power(x, 0.5))
# [1.         1.41421356 1.73205081 2.        ]

y = np.square(x)
print(y)
# [ 1  4  9 16]
print(np.power(x, 2))
# [ 1  4  9 16]
```

------

### 三角函数

#### numpy.sin

#### numpy.cos

#### numpy.tan

#### numpy.arcsin

#### numpy.arccos

#### numpy.arctan

**通用函数**（universal function）通常叫做 ufunc，它对数组中的各个元素逐一进行操作。这表明，通用函数分别处理输入数组的每个元素，生成的结果组成一个新的输出数组。输出数组的大小跟输入数组相同。<br/>

三角函数等很多数学运算符合通用函数的定义，例如，计算平方根的 `sqrt()` 函数、用来取对数的 `log()` 函数和求正弦值的 `sin()` 函数。<br/>

【例】

```python
import numpy as np

x = np.linspace(start=0, stop=np.pi / 2, num=10)
print(x)
# [0.         0.17453293 0.34906585 0.52359878 0.6981317  0.87266463
#  1.04719755 1.22173048 1.3962634  1.57079633]

y = np.sin(x)
print(y)
# [0.         0.17364818 0.34202014 0.5        0.64278761 0.76604444
#  0.8660254  0.93969262 0.98480775 1.        ]

z = np.arcsin(y)
print(z)
# [0.         0.17453293 0.34906585 0.52359878 0.6981317  0.87266463
#  1.04719755 1.22173048 1.3962634  1.57079633]

y = np.cos(x)
print(y)
# [1.00000000e+00 9.84807753e-01 9.39692621e-01 8.66025404e-01
#  7.66044443e-01 6.42787610e-01 5.00000000e-01 3.42020143e-01
#  1.73648178e-01 6.12323400e-17]

z = np.arccos(y)
print(z)
# [0.         0.17453293 0.34906585 0.52359878 0.6981317  0.87266463
#  1.04719755 1.22173048 1.3962634  1.57079633]

y = np.tan(x)
print(y)
# [0.00000000e+00 1.76326981e-01 3.63970234e-01 5.77350269e-01
#  8.39099631e-01 1.19175359e+00 1.73205081e+00 2.74747742e+00
#  5.67128182e+00 1.63312394e+16]

z = np.arctan(y)
print(z)
# [0.         0.17453293 0.34906585 0.52359878 0.6981317  0.87266463
#  1.04719755 1.22173048 1.3962634  1.57079633]
```

------

### 指数和对数

#### numpy.exp

#### numpy.log

#### numpy.exp2

#### numpy.log2

#### numpy.log10

【例】The natural logarithm `log` is the inverse of the exponential function, so that `log(exp(x)) = x`. The natural logarithm is logarithm in base `e`.

```python
import numpy as np

x = np.arange(1, 5)
print(x)
# [1 2 3 4]
y = np.exp(x)
print(y)
# [ 2.71828183  7.3890561  20.08553692 54.59815003]
z = np.log(y)
print(z)
# [1. 2. 3. 4.]
```

------

### 加法函数、乘法函数

#### numpy.sum

- `numpy.sum(a[, axis=None, dtype=None, out=None, ...])`：Sum of array elements over a given axis.

通过不同的 `axis`，numpy 会沿着不同的方向进行操作；如果不设置，那么对所有的元素操作；如果 `axis=0`，则沿着纵轴进行操作；`axis=1`，则沿着横轴进行操作。但这只是简单的二维数组，如果是多维的呢？可以总结为一句话：设 `axis=i`，则 numpy 沿着第 `i` 个下标变化的方向进行操作。

#### numpy.cumsum

- `numpy.cunsum(a, axis=None, dtype=None, out=None)`：Return the cumulative sum of the elements along a given axis.

**聚合函数**是指对一组值（比如一个数组）进行操作，返回一个单一值作为结果的函数。因而，求数组所有元素之和的函数就是聚合函数。`ndarray` 类实现了多个这样的函数。<br/>

【例】返回给定轴上的数组元素的总和。

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.sum(x)
print(y)  # 575

y = np.sum(x, axis=0)
print(y)  # [105 110 115 120 125]

y = np.sum(x, axis=1)
print(y)  # [ 65  90 115 140 165]
```

【例】返回给定轴上的数组元素的累加和。

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.cumsum(x)
print(y)
# [ 11  23  36  50  65  81  98 116 135 155 176 198 221 245 270 296 323 351
#  380 410 441 473 506 540 575]

y = np.cumsum(x, axis=0)
print(y)
# [[ 11  12  13  14  15]
#  [ 27  29  31  33  35]
#  [ 48  51  54  57  60]
#  [ 74  78  82  86  90]
#  [105 110 115 120 125]]

y = np.cumsum(x, axis=1)
print(y)
# [[ 11  23  36  50  65]
#  [ 16  33  51  70  90]
#  [ 21  43  66  90 115]
#  [ 26  53  81 110 140]
#  [ 31  63  96 130 165]]
```

#### numpy.prod 乘积

#### numpy.cumprod 累乘

【例】返回给定轴上数组元素的乘积。

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.prod(x)
print(y)  # 788529152

y = np.prod(x, axis=0)
print(y)
# [2978976 3877632 4972968 6294624 7875000]

y = np.prod(x, axis=1)
print(y)
# [  360360  1860480  6375600 17100720 38955840]
```

【例】返回给定轴上数组元素的累乘。

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.cumprod(x)
print(y)
# [         11         132        1716       24024      360360     5765760
#     98017920  1764322560  -837609728   427674624   391232512    17180672
#    395155456   893796352   870072320  1147043840   905412608  -418250752
#    755630080  1194065920 -1638662144  -897581056   444596224 -2063597568
#    788529152]

y = np.cumprod(x, axis=0)
print(y)
# [[     11      12      13      14      15]
#  [    176     204     234     266     300]
#  [   3696    4488    5382    6384    7500]
#  [  96096  121176  150696  185136  225000]
#  [2978976 3877632 4972968 6294624 7875000]]

y = np.cumprod(x, axis=1)
print(y)
# [[      11      132     1716    24024   360360]
#  [      16      272     4896    93024  1860480]
#  [      21      462    10626   255024  6375600]
#  [      26      702    19656   570024 17100720]
#  [      31      992    32736  1113024 38955840]]
```

#### numpy.diff 差值

- `numpy.diff(a, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue)`：Calculate the n-th discrete difference along the given recursively.
  - `a`：输入矩阵
  - `n`：可选，代表要执行几次差值
  - `axis`：默认是最后一个

The first difference is given by `out[i] = a[i+1] - a[i]` along the given axis, higher differences are calculated by using `diff` recursively.<br/>

【例】沿着给定轴计算第 N 维的离散差值

```python
import numpy as np

A = np.arange(2, 14).reshape((3, 4))
A[1, 1] = 8
print(A)
# [[ 2  3  4  5]
#  [ 6  8  8  9]
#  [10 11 12 13]]
print(np.diff(A))
# [[1 1 1]
#  [2 0 1]
#  [1 1 1]]
print(np.diff(A, axis=0))
# [[4 5 4 4]
#  [4 3 4 4]]
```

------

### 四舍五入

#### numpy.around 舍入

- `numpy.around(a, decimals=0, out=None)`：Evenly round to the given number of decimals.

【例】将数组舍入到给定的小数位数。

```python
import numpy as np

x = np.random.rand(3, 3) * 10
print(x)
# [[6.59144457 3.78566113 8.15321227]
#  [1.68241475 3.78753332 7.68886328]
#  [2.84255822 9.58106727 7.86678037]]

y = np.around(x)
print(y)
# [[ 7.  4.  8.]
#  [ 2.  4.  8.]
#  [ 3. 10.  8.]]

y = np.around(x, decimals=2)
print(y)
# [[6.59 3.79 8.15]
#  [1.68 3.79 7.69]
#  [2.84 9.58 7.87]]
```

#### numpy.ceil 上限

#### numpy.floor 下限

【例】

```python
import numpy as np

x = np.random.rand(3, 3) * 10
print(x)
# [[0.67847795 1.33073923 4.53920122]
#  [7.55724676 5.88854047 2.65502046]
#  [8.67640444 8.80110812 5.97528726]]

y = np.ceil(x)
print(y)
# [[1. 2. 5.]
#  [8. 6. 3.]
#  [9. 9. 6.]]

y = np.floor(x)
print(y)
# [[0. 1. 4.]
#  [7. 5. 2.]
#  [8. 8. 5.]]
```

------

### 其它

#### numpy.clip 裁剪

- `numpy.clip(a, a_min, a_max, out=None, **kwargs)`：Clip(limit) the values in an array.

Given an interal, values outside the interal are clipped to the interal edges. For example, if an interval of `[0, 1]` is specified, values smaller than 0 become 0, and values larger than 1 become 1.<br/>

【例】裁剪（限制）数组中的值。

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.clip(x, a_min=20, a_max=30)
print(y)
# [[20 20 20 20 20]
#  [20 20 20 20 20]
#  [21 22 23 24 25]
#  [26 27 28 29 30]
#  [30 30 30 30 30]]
```

#### numpy.absolute 绝对值

#### numpy.abs

【例】

```python
import numpy as np

x = np.arange(-5, 5)
print(x)
# [-5 -4 -3 -2 -1  0  1  2  3  4]

y = np.abs(x)
print(y)
# [5 4 3 2 1 0 1 2 3 4]

y = np.absolute(x)
print(y)
# [5 4 3 2 1 0 1 2 3 4]
```

#### numpy.sign 返回数字符号的逐元素指示

【例】

```python
x = np.arange(-5, 5)
print(x)
# [-5 -4 -3 -2 -1  0  1  2  3  4]
print(np.sign(x))
# [-1 -1 -1 -1 -1  0  1  1  1  1]
```

------

## 【练习】

### 数学函数

#### 1.

- stem

将数组 a 中大于 30 的值替换为 30，小于 10 的值替换为 10。

```python
a = np.random.uniform(1, 50, 20)
```

- code

```python
import numpy as np

np.set_printoptions(precision=2)
np.random.seed(100)
a = np.random.uniform(1, 50, 20)
print(a)
# [27.63 14.64 21.8  42.39  1.23  6.96 33.87 41.47  7.7  29.18 44.67 11.25
#  10.08  6.31 11.77 48.95 40.77  9.43 41.   14.43]

# 方法一
b = np.clip(a, a_min=10, a_max=30)
print(b)
# [27.63 14.64 21.8  30.   10.   10.   30.   30.   10.   29.18 30.   11.25
#  10.08 10.   11.77 30.   30.   10.   30.   14.43]

# 方法二
b = np.where(a < 10, 10, a)
b = np.where(b > 30, 30, b)
print(b)
# [27.63 14.64 21.8  30.   10.   10.   30.   30.   10.   29.18 30.   11.25
#  10.08 10.   11.77 30.   30.   10.   30.   14.43]
```

#### 2.

- stem

找到一个一维数字数组 a 中的所有峰值。峰顶是两边被较小数值包围的点。

```python
a = np.array([1, 3, 7, 1, 2, 6, 0, 1])
```

- code

```python
import numpy as np

a = np.array([1, 3, 7, 1, 2, 6, 0, 1])
b1 = np.diff(a)
b2 = np.sign(b1)
b3 = np.diff(b2)

print(b1)  # [ 2  4 -6  1  4 -6  1]
print(b2)  # [ 1  1 -1  1  1 -1  1]
print(b3)  # [ 0 -2  2  0 -2  2]
index = np.where(np.equal(b3, -2))[0] + 1
print(index)  # [2 5]
```

#### 3.

- stem

对于给定的一维数组，计算窗口大小为 3 的移动平均值。

```python
z = np.random.randint(10, size=10)
```

- code

```python
import numpy as np

np.random.seed(100)
z = np.random.randint(10, size=10)
print(z)


# [8 8 3 7 7 0 4 2 5 2]

def MovingAverage(arr, n=3):
    a = np.cumsum(arr)
    a[n:] = a[n:] - a[:-n]
    return a[n - 1:] / n


r = MovingAverage(z, 3)
print(np.around(r, 2))
# [6.33 6.   5.67 4.67 3.67 2.   3.67 3.  ]
```

#### 4.

- stem

对一个 5 $\times$ 5 的矩阵做归一化

- code

```python
import numpy as np

Z = np.random.random((5, 5))
Zmax, Zmin = Z.max(), Z.min()
Z = (Z - Zmin) / (Zmax - Zmin)
print(Z)
```

#### 5.

- stem

用**五种**不同的方法去提取一个随机数组的整数部分

- code

```python
import numpy as np

Z = np.random.uniform(0,10,10)
# 方法一
print (Z - Z%1)
# 方法二
print (np.floor(Z))
# 方法三
print (np.ceil(Z)-1)
# 方法四
print (Z.astype(int))
# 方法五
print (np.trunc(Z))
```

#### 6.

- stem

考虑一维数组 Z，构建一个二维数组，其第一行为 (Z[0], Z[1], Z[2])，随后的每一行都移位 1（最后一行应为 (Z[-3], Z[-2], Z[-1])。

- code

```python
import numpy as np
from numpy.lib import stride_tricks

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)
print(Z)
```

#### 7.

- stem

考虑两组点集 $P_0$ 和 $P_1$ 去描述一组线（二维）和一个点 $p$，如何计算点 $p$ 到每一条线 $i$ （$P_0[i]$，$P_1[i]$）的距离？

- code

```python
import numpy as np


def distance(P0, P1, p):
    A = -1 / (P1[:, 0] - P0[:, 0])
    B = 1 / (P1[:, 1] - P0[:, 1])
    C = P0[:, 0] / (P1[:, 0] - P0[:, 0]) - P0[:, 1] / (P1[:, 1] - P0[:, 1])
    return np.abs(A * p[:, 0] + B * p[:, 1] + C) / np.sqrt(A ** 2 + B ** 2)


P0 = np.random.uniform(-10, 10, (10, 2))
P1 = np.random.uniform(-10, 10, (10, 2))
p = np.random.uniform(-10, 10, (1, 2))

print(distance(P0, P1, p))
```

#### 8.

- stem

画正弦函数和余弦函数，x = np.arange(0, 3 * np.pi, 0.1)

- code

```python
import numpy as np
from matplotlib import pyplot as plt

x = np.arange(0, 3 * np.pi, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1)
plt.plot(x, y2)
plt.show()
```

![正余弦函数](https://gitee.com/Miraclezjy/utoolspic/raw/master/pic3-2022-1-1621:47:45.png)

#### 9.

- stem

减去矩阵每一行的平均值

- code

```python
import numpy as np

X = np.random.rand(5, 10)
# 新
Y = X - X.mean(axis=1, keepdims=True)
# 旧
Y = X - X.mean(axis=1).reshape(-1, 1)
print(Y)
```

#### 10.

- stem

进行概率统计分析

```python
arr1 = np.random.randint(1,10,10)
arr2 = np.random.randint(1,10,10)
```

- code

```python
import numpy as np

arr1 = np.random.randint(1, 10, 10)
arr2 = np.random.randint(1, 10, 10)

print("arr1的平均数为:%s" % np.mean(arr1))
print("arr1的中位数为:%s" % np.median(arr1))
print("arr1的方差为:%s" % np.var(arr1))
print("arr1的标准差为:%s" % np.std(arr1))
print("arr1,arr的相关性矩阵为:%s" % np.cov(arr1, arr2))
print("arr1,arr的协方差矩阵为:%s" % np.corrcoef(arr1, arr2))
```

### 逻辑函数

#### 1.

- stem

获取 a 和 b 元素匹配的位置

```python
a = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])
b = np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8])
```

- code

```python
import numpy as np

a = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])
b = np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8])
mask = np.equal(a, b)

# 方法一
x = np.where(mask)
print(x)  # (array([1, 3, 5, 7], dtype=int64),)

# 方法二
x = np.nonzero(mask)
print(x)  # (array([1, 3, 5, 7], dtype=int64),)
```

#### 2.

- stem

获取 5 到 10 之间的所有元素

```python
a = np.array([2, 6, 1, 9, 10, 3, 27])
```

- code

```python
import numpy as np

a = np.array([2, 6, 1, 9, 10, 3, 27])
mask = np.logical_and(np.greater_equal(a, 5), np.less_equal(a, 10))

# 方法一
x = np.where(mask)
print(a[x])  # [ 6  9 10]

# 方法二
x = np.nonzero(mask)
print(a[x])  # [ 6  9 10]

# 方法三
x = a[np.logical_and(a >= 5, a <= 10)]
print(x)  # [ 6  9 10]
```

#### 3.

- stem

对于两个随机数组 A 和 B，检查它们是否相等

- code

```python
import numpy as np

A = np.random.randint(0, 2, 5)
B = np.random.randint(0, 2, 5)
# Assuming identical shape of the arrays and a tolerance for the comparison of values
equal = np.allclose(A, B)
print(equal)
```

#### 4.

- stem

如何对布尔值取反，或者原位（in-place）改变浮点数的符号（sign）？

- code

```python
import numpy as np

Z = np.array([0, 1])
print(Z)
np.logical_not(Z, out=Z)
# Z = np.random.uniform(-1.0,1.0,100)

# np.negative(Z, out=Z)
Z = np.array([0.2, 1.15])
print(Z)
np.negative(Z, out=Z)
```

#### 5.

- stem

找出数组中与给定值最接近的数

- code

```python
import numpy as np

Z = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
print(Z)
z = 5.1
np.abs(Z - z).argmin()
print(Z.flat[np.abs(Z - z).argmin()])
```

# 第五章 排序搜索计数及集合操作

## 一、排序，搜索和计数

### 排序

#### numpy.sort()

- `numpy.sort(a[, axis=-1, kind='quicksort', order=None])`：Return a sorted **copy** of an array.
  - `axis`：排序沿数组的（轴）方向， 0 表示按列，1 表示按行，None 表示展开来排序，默认为 -1，表示沿最后的轴排序。
  - `kind`：排序的算法，提供了快排 `quicksort`、混排 `mergesort`、堆排 `heapsort`，默认为 `quicksort`。
  - `order`：排序的字段名，可指定字段排序，默认为 None。

【例】

```python
import numpy as np

np.random.seed(20200612)
x = np.random.rand(5, 5) * 10
x = np.around(x, 2)
print(x)
# [[2.32 7.54 9.78 1.73 6.22]
#  [6.93 5.17 9.28 9.76 8.25]
#  [0.01 4.23 0.19 1.73 9.27]
#  [7.99 4.97 0.88 7.32 4.29]
#  [9.05 0.07 8.95 7.9  6.99]]

y = np.sort(x)
print(y)
# [[1.73 2.32 6.22 7.54 9.78]
#  [5.17 6.93 8.25 9.28 9.76]
#  [0.01 0.19 1.73 4.23 9.27]
#  [0.88 4.29 4.97 7.32 7.99]
#  [0.07 6.99 7.9  8.95 9.05]]

y = np.sort(x, axis=0)
print(y)
# [[0.01 0.07 0.19 1.73 4.29]
#  [2.32 4.23 0.88 1.73 6.22]
#  [6.93 4.97 8.95 7.32 6.99]
#  [7.99 5.17 9.28 7.9  8.25]
#  [9.05 7.54 9.78 9.76 9.27]]

y = np.sort(x, axis=1)
print(y)
# [[1.73 2.32 6.22 7.54 9.78]
#  [5.17 6.93 8.25 9.28 9.76]
#  [0.01 0.19 1.73 4.23 9.27]
#  [0.88 4.29 4.97 7.32 7.99]
#  [0.07 6.99 7.9  8.95 9.05]]
```

【例】

```python
import numpy as np

dt = np.dtype([('name', 'S10'), ('age', np.int)])
a = np.array([("Mike", 21), ("Nancy", 25), ("Bob", 17), ("Jane", 27)], dtype=dt)
b = np.sort(a, order='name')
print(b)
# [(b'Bob', 17) (b'Jane', 27) (b'Mike', 21) (b'Nancy', 25)]

b = np.sort(a, order='age')
print(b)
# [(b'Bob', 17) (b'Mike', 21) (b'Nancy', 25) (b'Jane', 27)]
```

如果排序后，想用元素的索引位置替代排序后的实际结果，该怎么办呢？

#### numpy.argsort()

- `numpy.argsort(a[, axis=-1, kind='quicksort', order=None])`：Returns the indices that would sort an array.

【例】对数组沿给定轴执行间接排序，并使用指定排序类型返回数据的索引数组。这个索引数组用于构造排序后的数组。

```python
import numpy as np

np.random.seed(20200612)
x = np.random.randint(0, 10, 10)
print(x)
# [6 1 8 5 5 4 1 2 9 1]

y = np.argsort(x)
print(y)
# [1 6 9 7 5 3 4 0 2 8]

print(x[y])
# [1 1 1 2 4 5 5 6 8 9]

y = np.argsort(-x)
print(y)
# [8 2 0 3 4 5 7 1 6 9]

print(x[y])
# [9 8 6 5 5 4 2 1 1 1]
```

【例】

```python
import numpy as np

np.random.seed(20200612)
x = np.random.rand(5, 5) * 10
x = np.around(x, 2)
print(x)
# [[2.32 7.54 9.78 1.73 6.22]
#  [6.93 5.17 9.28 9.76 8.25]
#  [0.01 4.23 0.19 1.73 9.27]
#  [7.99 4.97 0.88 7.32 4.29]
#  [9.05 0.07 8.95 7.9  6.99]]

y = np.argsort(x)
print(y)
# [[3 0 4 1 2]
#  [1 0 4 2 3]
#  [0 2 3 1 4]
#  [2 4 1 3 0]
#  [1 4 3 2 0]]

y = np.argsort(x, axis=0)
print(y)
# [[2 4 2 0 3]
#  [0 2 3 2 0]
#  [1 3 4 3 4]
#  [3 1 1 4 1]
#  [4 0 0 1 2]]

y = np.argsort(x, axis=1)
print(y)
# [[3 0 4 1 2]
#  [1 0 4 2 3]
#  [0 2 3 1 4]
#  [2 4 1 3 0]
#  [1 4 3 2 0]]

y = np.array([np.take(x[i], np.argsort(x[i])) for i in range(5)])
# numpy.take(a, indices, axis=None, out=None, mode='raise')沿轴从数组中获取元素。
print(y)
# [[1.73 2.32 6.22 7.54 9.78]
#  [5.17 6.93 8.25 9.28 9.76]
#  [0.01 0.19 1.73 4.23 9.27]
#  [0.88 4.29 4.97 7.32 7.99]
#  [0.07 6.99 7.9  8.95 9.05]]
```

如何将数据按照某一指标进行排序呢？

#### numpy.lexsort()

- `numpy.lexsort(keys[, axis=-1])`：Perform an indirect stable sort using a sequence of keys.（使用键序列执行间接稳定排序）
- 给定多个可以在电子表格中解释为列的排序键，`lexsort` 返回一个整数索引数组，该数组描述了按多个列排序的顺序。序列中的最后一个键用于主排序顺序，倒数第二个键用于辅助排序顺序，依此类推。`keys` 参数必须是可以转换为相同形状的数组的对象序列。如果为 `keys` 参数提供了 2D 数组，则将其行解释为排序键，并根据最后一行，倒数第二行等进行排序。

【例】按照第一列的升序或者降序对整体数据进行排序。

```python
import numpy as np

np.random.seed(20200612)
x = np.random.rand(5, 5) * 10
x = np.around(x, 2)
print(x)
# [[2.32 7.54 9.78 1.73 6.22]
#  [6.93 5.17 9.28 9.76 8.25]
#  [0.01 4.23 0.19 1.73 9.27]
#  [7.99 4.97 0.88 7.32 4.29]
#  [9.05 0.07 8.95 7.9  6.99]]

index = np.lexsort([x[:, 0]])
print(index)
# [2 0 1 3 4]

y = x[index]
print(y)
# [[0.01 4.23 0.19 1.73 9.27]
#  [2.32 7.54 9.78 1.73 6.22]
#  [6.93 5.17 9.28 9.76 8.25]
#  [7.99 4.97 0.88 7.32 4.29]
#  [9.05 0.07 8.95 7.9  6.99]]

index = np.lexsort([-1 * x[:, 0]])
print(index)
# [4 3 1 0 2]

y = x[index]
print(y)
# [[9.05 0.07 8.95 7.9  6.99]
#  [7.99 4.97 0.88 7.32 4.29]
#  [6.93 5.17 9.28 9.76 8.25]
#  [2.32 7.54 9.78 1.73 6.22]
#  [0.01 4.23 0.19 1.73 9.27]]
```

【例】

```python
import numpy as np

x = np.array([1, 5, 1, 4, 3, 4, 4])
y = np.array([9, 4, 0, 4, 0, 2, 1])
a = np.lexsort([x])
b = np.lexsort([y])
print(a)
# [0 2 4 3 5 6 1]
print(x[a])
# [1 1 3 4 4 4 5]

print(b)
# [2 4 6 5 1 3 0]
print(y[b])
# [0 0 1 2 4 4 9]

z = np.lexsort([y, x])
print(z)
# [2 0 4 6 5 3 1]
print(x[z])
# [1 1 3 4 4 4 5]

z = np.lexsort([x, y])
print(z)
# [2 4 6 5 3 1 0]
print(y[z])
# [0 0 1 2 4 4 9]
```

#### numpy.partition()

- `numpy.partition(a, kth, axis=-1, kind='introselect', order=None)`：Return a partitioned copy of an array.

Creates a copy of the array with its elements rearranged in such a way that the value of the element in k-th position is in the position it would be in a sorted array. All elements smaller than the k-th element are moved before this element and all equal or greater are moved behind it. The ordering of the elements in the two partitions is undefined.

【例】以索引是 kth 的元素为基准，将元素分成两部分，即大于该元素的放在其后面，小于该元素的放在其前面，这里有点类似于快排。

```python
import numpy as np

np.random.seed(100)
x = np.random.randint(1, 30, [8, 3])
print(x)
# [[ 9 25  4]
#  [ 8 24 16]
#  [17 11 21]
#  [ 3 22  3]
#  [ 3 15  3]
#  [18 17 25]
#  [16  5 12]
#  [29 27 17]]

y = np.sort(x, axis=0)
print(y)
# [[ 3  5  3]
#  [ 3 11  3]
#  [ 8 15  4]
#  [ 9 17 12]
#  [16 22 16]
#  [17 24 17]
#  [18 25 21]
#  [29 27 25]]

z = np.partition(x, kth=2, axis=0)
print(z)
# [[ 3  5  3]
#  [ 3 11  3]
#  [ 8 15  4]
#  [ 9 22 21]
#  [17 24 16]
#  [18 17 25]
#  [16 25 12]
#  [29 27 17]]
```

【例】选取每一列第三小的数据。

```python
import numpy as np

np.random.seed(100)
x = np.random.randint(1, 30, [8, 3])
print(x)
# [[ 9 25  4]
#  [ 8 24 16]
#  [17 11 21]
#  [ 3 22  3]
#  [ 3 15  3]
#  [18 17 25]
#  [16  5 12]
#  [29 27 17]]
z = np.partition(x, kth=2, axis=0)
print(z[2])
# [ 8 15  4]
```

【例】选取每一列第三大的数据。

```python
import numpy as np

np.random.seed(100)
x = np.random.randint(1, 30, [8, 3])
print(x)
# [[ 9 25  4]
#  [ 8 24 16]
#  [17 11 21]
#  [ 3 22  3]
#  [ 3 15  3]
#  [18 17 25]
#  [16  5 12]
#  [29 27 17]]
z = np.partition(x, kth=-3, axis=0)
print(z[-3])
# [17 24 17]
```

#### numpy.argpartition()

- `numpy.argpartition(a, kth, axis=-1, kind='introselect', order=None)`

【例】

```python
import numpy as np

np.random.seed(100)
x = np.random.randint(1, 30, [8, 3])
print(x)
# [[ 9 25  4]
#  [ 8 24 16]
#  [17 11 21]
#  [ 3 22  3]
#  [ 3 15  3]
#  [18 17 25]
#  [16  5 12]
#  [29 27 17]]

y = np.argsort(x, axis=0)
print(y)
# [[3 6 3]
#  [4 2 4]
#  [1 4 0]
#  [0 5 6]
#  [6 3 1]
#  [2 1 7]
#  [5 0 2]
#  [7 7 5]]

z = np.argpartition(x, kth=2, axis=0)
print(z)
# [[3 6 3]
#  [4 2 4]
#  [1 4 0]
#  [0 3 2]
#  [2 1 1]
#  [5 5 5]
#  [6 0 6]
#  [7 7 7]]
```

【例】选取每一列第三小的数的索引。

```python
import numpy as np

np.random.seed(100)
x = np.random.randint(1, 30, [8, 3])
print(x)
# [[ 9 25  4]
#  [ 8 24 16]
#  [17 11 21]
#  [ 3 22  3]
#  [ 3 15  3]
#  [18 17 25]
#  [16  5 12]
#  [29 27 17]]

z = np.argpartition(x, kth=2, axis=0)
print(z[2])
# [1 4 0]
```

【例】选取每一列第三大的数的索引。

```python
import numpy as np

np.random.seed(100)
x = np.random.randint(1, 30, [8, 3])
print(x)
# [[ 9 25  4]
#  [ 8 24 16]
#  [17 11 21]
#  [ 3 22  3]
#  [ 3 15  3]
#  [18 17 25]
#  [16  5 12]
#  [29 27 17]]

z = np.argpartition(x, kth=-3, axis=0)
print(z[-3])
# [2 1 7]
```

------

### 搜索

#### numpy.argmax()

- `numpy.argmax(a[, axis=None, out=None])`：Returns the indices of the maximum values along an axis.

【例】

```python
import numpy as np

np.random.seed(20200612)
x = np.random.rand(5, 5) * 10
x = np.around(x, 2)
print(x)
# [[2.32 7.54 9.78 1.73 6.22]
#  [6.93 5.17 9.28 9.76 8.25]
#  [0.01 4.23 0.19 1.73 9.27]
#  [7.99 4.97 0.88 7.32 4.29]
#  [9.05 0.07 8.95 7.9  6.99]]

y = np.argmax(x)
print(y)  # 2

y = np.argmax(x, axis=0)
print(y)
# [4 0 0 1 2]

y = np.argmax(x, axis=1)
print(y)
# [2 3 4 0 0]
```

#### numpy.argmin()

#### numpy.nonzero()

- `numpy.nonzero(a)`：Return the indices of the elements that are non-zero.

其值为非零元素的下标在对应轴上的值。

1. 只有 a 中非零元素才会有索引值，那些零值元素没有索引值；
2. 返回一个长度为 a.ndim 的元组（tuple），元组的每个元素都是一个整数数组（array）；
3. 每一个 array 都是从一个维度上来描述其索引值。比如，如果 a 是一个二维数组，则 tuple 包含两个 array，第一个 array 从行维度来描述索引值，第二个 array 从列角度来描述索引值；
4. `np.transpose(np.nonzero(x))` 函数能够描述出每一个非零元素在不同维度的索引值；
5. 通过 `a[nonzero(a)]` 得到所有 a 中的非零值。

【例】一维数组

```python
import numpy as np

x = np.array([0, 2, 3])
print(x)  # [0 2 3]
print(x.shape)  # (3,)
print(x.ndim)  # 1

y = np.nonzero(x)
print(y)  # (array([1, 2], dtype=int64),)
print(np.array(y))  # [[1 2]]
print(np.array(y).shape)  # (1, 2)
print(np.array(y).ndim)  # 2
print(np.transpose(y))
# [[1]
#  [2]]
print(x[np.nonzero(x)])
#[2, 3]
```

【例】二维数组

```python
import numpy as np

x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
print(x)
# [[3 0 0]
#  [0 4 0]
#  [5 6 0]]
print(x.shape)  # (3, 3)
print(x.ndim)  # 2

y = np.nonzero(x)
print(y)
# (array([0, 1, 2, 2], dtype=int64), array([0, 1, 0, 1], dtype=int64))
print(np.array(y))
# [[0 1 2 2]
#  [0 1 0 1]]
print(np.array(y).shape)  # (2, 4)
print(np.array(y).ndim)  # 2

y = x[np.nonzero(x)]
print(y)  # [3 4 5 6]

y = np.transpose(np.nonzero(x))
print(y)
# [[0 0]
#  [1 1]
#  [2 0]
#  [2 1]]
```

【例】三维数组

```python
import numpy as np

x = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]], [[0, 0], [1, 0]]])
print(x)
# [[[0 1]
#   [1 0]]
#
#  [[0 1]
#   [1 0]]
#
#  [[0 0]
#   [1 0]]]
print(np.shape(x))  # (3, 2, 2)
print(x.ndim)  # 3

y = np.nonzero(x)
print(np.array(y))
# [[0 0 1 1 2]
#  [0 1 0 1 1]
#  [1 0 1 0 0]]
print(np.array(y).shape)  # (3, 5)
print(np.array(y).ndim)  # 2
print(y)
# (array([0, 0, 1, 1, 2], dtype=int64), array([0, 1, 0, 1, 1], dtype=int64), array([1, 0, 1, 0, 0], dtype=int64))
print(x[np.nonzero(x)])
#[1 1 1 1 1]
```

【例】`nonzero()` 将布尔数组转换成整数数组进行操作。

```python
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x)
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

y = x > 3
print(y)
# [[False False False]
#  [ True  True  True]
#  [ True  True  True]]

y = np.nonzero(x > 3)
print(y)
# (array([1, 1, 1, 2, 2, 2], dtype=int64), array([0, 1, 2, 0, 1, 2], dtype=int64))

y = x[np.nonzero(x > 3)]
print(y)
# [4 5 6 7 8 9]

y = x[x > 3]
print(y)
# [4 5 6 7 8 9]
```

#### numpy.where()

- `numpy.where(condition, [x=None, y=None])`：Return elements chosen from x or y depending on condition.（有点像 `C` 中的三目运算符 `?:`）

【例】满足条件 `condition`，输出 `x`，不满足输出 `y`。

```python
import numpy as np

x = np.arange(10)
print(x)
# [0 1 2 3 4 5 6 7 8 9]

y = np.where(x < 5, x, 10 * x)
print(y)
# [ 0  1  2  3  4 50 60 70 80 90]

x = np.array([[0, 1, 2],
              [0, 2, 4],
              [0, 3, 6]])
y = np.where(x < 4, x, -1)
print(y)
# [[ 0  1  2]
#  [ 0  2 -1]
#  [ 0  3 -1]]
```

【例】只有 `condition`，没有 `x` 和 `y`，则输出满足条件（即非 0）元素的坐标（等价于 `numpy.nonzero`）。这里的坐标以 tuple 的形式给出，通常原数组有多少维，输出的 tuple 中就包含几个数组，分别对应符合条件元素的各维坐标。

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.where(x > 5)
print(y)
# (array([5, 6, 7], dtype=int64),)
print(x[y])
# [6 7 8]

y = np.nonzero(x > 5)
print(y)
# (array([5, 6, 7], dtype=int64),)
print(x[y])
# [6 7 8]

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.where(x > 25)
print(y)
# (array([3, 3, 3, 3, 3, 4, 4, 4, 4, 4], dtype=int64), array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=int64))

print(x[y])
# [26 27 28 29 30 31 32 33 34 35]

y = np.nonzero(x > 25)
print(y)
# (array([3, 3, 3, 3, 3, 4, 4, 4, 4, 4], dtype=int64), array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=int64))
print(x[y])
# [26 27 28 29 30 31 32 33 34 35]
```

#### numpy.searchsorted()

- `numpy.searchsorted(a, v[, side='left', sorter=None]`：Find indices where elements should be inserted to maintain order.
  - `a`：一维输入数组。但 `sorter` 参数为 None 时，`a` 必须为升序数组；否则，`sorter` 不能为空，存放 `a` 中元素的 `index`，用于反映 `a` 数组的升序排列方式。
  - `v`：插入 `a` 数组的值，可以为单个元素，`list` 或 `ndarray`。
  - `side`：查询方向，当为 `left` 时，将返回第一个符合条件的元素下标；当为 `right` 时，将返回最后一个符合条件的元素下标。
  - `sorter`：一维数组存放 `a` 数组元素的 `index`，`index` 对应元素为升序。

【例】

```python
import numpy as np

x = np.array([0, 1, 5, 9, 11, 18, 26, 33])
y = np.searchsorted(x, 15)
print(y)  # 5

y = np.searchsorted(x, 15, side='right')
print(y)  # 5

y = np.searchsorted(x, -1)
print(y)  # 0

y = np.searchsorted(x, -1, side='right')
print(y)  # 0

y = np.searchsorted(x, 35)
print(y)  # 8

y = np.searchsorted(x, 35, side='right')
print(y)  # 8

y = np.searchsorted(x, 11)
print(y)  # 4

y = np.searchsorted(x, 11, side='right')
print(y)  # 5

y = np.searchsorted(x, 0)
print(y)  # 0

y = np.searchsorted(x, 0, side='right')
print(y)  # 1

y = np.searchsorted(x, 33)
print(y)  # 7

y = np.searchsorted(x, 33, side='right')
print(y)  # 8
```

【例】

```python
import numpy as np

x = np.array([0, 1, 5, 9, 11, 18, 26, 33])
y = np.searchsorted(x, [-1, 0, 11, 15, 33, 35])
print(y)  # [0 0 4 5 7 8]

y = np.searchsorted(x, [-1, 0, 11, 15, 33, 35], side='right')
print(y)  # [0 1 5 5 8 8]
```

【例】

```python
import numpy as np

x = np.array([0, 1, 5, 9, 11, 18, 26, 33])
np.random.shuffle(x)
print(x)  # [33  1  9 18 11 26  0  5]

x_sort = np.argsort(x)
print(x_sort)  # [6 1 7 2 4 3 5 0]

y = np.searchsorted(x, [-1, 0, 11, 15, 33, 35], sorter=x_sort)
print(y)  # [0 0 4 5 7 8]

y = np.searchsorted(x, [-1, 0, 11, 15, 33, 35], side='right', sorter=x_sort)
print(y)  # [0 1 5 5 8 8]
```

------

### 计数

#### numpy.count_nonzero()

- `numpy.count_nonzero(a, axis=None)`：Counts the number of non-zero values in the array a.

【例】返回数组中的非 0 元素个数。

```python
import numpy as np

x = np.count_nonzero(np.eye(4))
print(x)  # 4

x = np.count_nonzero([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])
print(x)  # 5

x = np.count_nonzero([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]], axis=0)
print(x)  # [1 1 1 1 1]

x = np.count_nonzero([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]], axis=1)
print(x)  # [2 3]
```

## 二、集合操作

### 构造集合

- `numpy.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None)`：Find the unique elements of an array.
  - `return_index=True`：表示返回新列表元素在旧列表中的位置。
  - `return_inverse=True`：表示返回旧列表元素在新列表中的位置。
  - `return_counts=True`：表示返回新列表元素在旧列表中出现的次数。

【例】找出数组中的唯一值并返回已排序的结果。

```python
import numpy as np

x = np.unique([1, 1, 3, 2, 3, 3])
print(x)  # [1 2 3]

x = sorted(set([1, 1, 3, 2, 3, 3]))
print(x)  # [1, 2, 3]

x = np.array([[1, 1], [2, 3]])
u = np.unique(x)
print(u)  # [1 2 3]

x = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
y = np.unique(x, axis=0)
print(y)
# [[1 0 0]
#  [2 3 4]]

x = np.array(['a', 'b', 'b', 'c', 'a'])
u, index = np.unique(x, return_index=True)
print(u)  # ['a' 'b' 'c']
print(index)  # [0 1 3]
print(x[index])  # ['a' 'b' 'c']

x = np.array([1, 2, 6, 4, 2, 3, 2])
u, index = np.unique(x, return_inverse=True)
print(u)  # [1 2 3 4 6]
print(index)  # [0 1 4 3 1 2 1]
print(u[index])  # [1 2 6 4 2 3 2]

u, count = np.unique(x, return_counts=True)
print(u)  # [1 2 3 4 6]
print(count)  # [1 3 1 1 1]
```

### 布尔运算

- `numpy.in1d(ar1, ar2, assume_unique=False, invert=False)`：Test whether each element of a 1-D array is also present in a second array.

Returns a boolean array the same length as `ar1` that is True where an element of `ar1` is in `ar2` and False otherwise.

【例】前面的数组是否包含于后面的数组，返回布尔值。返回的值是针对第一个参数的数组的，所以维数和第一个参数一致，布尔值与数组的元素位置也一一对应。

```python
import numpy as np

test = np.array([0, 1, 2, 5, 0])
states = [0, 2]
mask = np.in1d(test, states)
print(mask)  # [ True False  True False  True]
print(test[mask])  # [0 2 0]

mask = np.in1d(test, states, invert=True)
print(mask)  # [False  True False  True False]
print(test[mask])  # [1 5]
```

#### 求两个集合的交集

- `numpy.intersect1d(ar1, ar2, assume_unique=False, return_indices=False)`：Find the intersection of two arrays.

Return the sorted, unique values that are in both of the input arrays.

【例】求两个数组的唯一化 + 求交集 + 排序函数。

```python
import numpy as np
from functools import reduce

x = np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])
print(x)  # [1 3]

x = np.array([1, 1, 2, 3, 4])
y = np.array([2, 1, 4, 6])
xy, x_ind, y_ind = np.intersect1d(x, y, return_indices=True)
print(x_ind)  # [0 2 4]
print(y_ind)  # [1 0 2]
print(xy)  # [1 2 4]
print(x[x_ind])  # [1 2 4]
print(y[y_ind])  # [1 2 4]

x = reduce(np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
print(x)  # [3]
```

#### 求两个集合的并集

- `numpy.union1d(ar1, ar2)`：Find the union of two array.

Return the unique, sorted array of values that are in either of the two input arrays.

【例】计算两个集合的并集，唯一化并排序

```python
import numpy as np
from functools import reduce

x = np.union1d([-1, 0, 1], [-2, 0, 2])
print(x)  # [-2 -1  0  1  2]
x = reduce(np.union1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
print(x)  # [1 2 3 4 6]
'''
functools.reduce(function, iterable[, initializer])
将两个参数的 function 从左至右积累地应用到 iterable 的条目，以便将该可迭代对象缩减为单一的值。 例如，reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) 是计算 ((((1+2)+3)+4)+5) 的值。 左边的参数 x 是积累值而右边的参数 y 则是来自 iterable 的更新值。 如果存在可选项 initializer，它会被放在参与计算的可迭代对象的条目之前，并在可迭代对象为空时作为默认值。 如果没有给出 initializer 并且 iterable 仅包含一个条目，则将返回第一项。

大致相当于：
def reduce(function, iterable, initializer=None):
    it = iter(iterable)
    if initializer is None:
        value = next(it)
    else:
        value = initializer
    for element in it:
        value = function(value, element)
    return value
'''
```

#### 求两个集合的差集

- `numpy.setdiffed(ar1, ar2, assume_unique=False)`：Find the set difference of two arrays.

Return the unique values in `ar1` that are not in `ar2`.

【例】集合的差，即元素存在于第一个函数不存在于第二个函数中。

```python
import numpy as np

a = np.array([1, 2, 3, 2, 4, 1])
b = np.array([3, 4, 5, 6])
x = np.setdiff1d(a, b)
print(x)  # [1 2]
```

#### 求两个集合的异或

- `numpy.setxor1d(ar1, ar2, assume_unique=False)`：Find the set exclusive-or of two arrays.

【例】集合的对称差，即两个集合的交集的补集。简而言之，就是两个数组中各自独立拥有的元素的集合。

```python
import numpy as np

a = np.array([1, 2, 3, 2, 4, 1])
b = np.array([3, 4, 5, 6])
x = np.setxor1d(a, b)
print(x)  # [1 2 5 6]
```

# 第六章 输入输出

## 一、输入和输出

### 1. numpy 二进制文件

`save()`、`savez() `和 `load()` 函数以 numpy 专用的二进制类型（`.npy`、`.npz`）保存和读取数据，这三个函数会自动处理ndim、dtype、shape 等信息，使用它们读写数组非常方便，但是 `save()` 和 `savez()` 输出的文件很难与其它语言编写的程序兼容。

【函数】

```python
def save(file, arr, allow_pickle=True, fix_imports=True):
    ...
```

- `save()` 函数：以 `.npy` 格式将数组保存到二进制文件中。
- `.npy` 格式：以二进制的方式存储文件，在二进制文件第一行以文本形式保存了数据的元信息（ndim、dtype、shape等），可以用二进制工具查看内容。

【函数】

```python
def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII'):
	...
```

- `load()` 函数：从 `.npy`、`.npz` 或 `pickled` 文件加载数组或 `pickled` 对象。
- `mmap_mode:{None, 'r+', 'r', 'w+', 'w', 'c'}`：读取文件的方式。
- `allow_pickle=False`：允许加载存储在 `.npy` 文件中的 `pickled` 对象数组。
- `fix_importings=True`：若为 True，pickle 将尝试将旧的 python2 名称映射到 python3 中使用的新名称。
- `encoding='ASCII'`：制定编码格式，默认为 “ASCII”。

【例】将一个数组保存到一个文件中。

```python
import numpy as np

outfile = r'.\test.npy'
np.random.seed(20200619)
x = np.random.uniform(low=0, high=1,size = [3, 5])
np.save(outfile, x)
y = np.load(outfile)
print(y)
# [[0.01123594 0.66790705 0.50212171 0.7230908  0.61668256]
#  [0.00668332 0.1234096  0.96092409 0.67925305 0.38596837]
#  [0.72342998 0.26258324 0.24318845 0.98795012 0.77370715]]
```

【函数】

```python
def savez(file, *args, **kwds):
    ...
```

- `savez() `函数：以未压缩的 `.npz` 格式将多个数组保存到单个文件中。
- `.npz` 格式：以压缩打包的方式存储文件，可以用压缩软件解压。
- `savez()` 函数：第一个参数是文件名，其后的参数都是需要保存的数组，也可以使用关键字参数为数组起一个名字，非关键字参数传递的数组会自动起名为`arr_0, arr_1, …`。
- `savez() `函数：输出的是一个压缩文件（扩展名为`.npz`），其中每个文件都是一个`save()`保存的 `.npy` 文件，文件名对应于数组名。`load()` 自动识别 `.npz` 文件，并且返回一个类似于字典的对象，可以通过数组名作为关键字获取数组的内容。

【例】将多个数组保存到一个文件。

```python
import numpy as np

outfile = r'.\test.npz'
x = np.linspace(0, np.pi, 5)
y = np.sin(x)
z = np.cos(x)
np.savez(outfile, x, y, z_d=z)
data = np.load(outfile)
np.set_printoptions(suppress=True)
print(data.files)  
# ['z_d', 'arr_0', 'arr_1']

print(data['arr_0'])
# [0.         0.78539816 1.57079633 2.35619449 3.14159265]

print(data['arr_1'])
# [0.         0.70710678 1.         0.70710678 0.        ]

print(data['z_d'])
# [ 1.          0.70710678  0.         -0.70710678 -1.        ]
```

用解压软件打开 test.npz 文件，会发现其中有三个文件：`arr_0.npy,arr_1.npy,z_d.npy`，其中分别保存着数组`x,y,z`的内容。

------

### 2. 文本文件

`savetxt()`，`loadtxt()`和`genfromtxt()`函数用来存储和读取文本文件（如`.TXT`，`.CSV`等）。`genfromtxt()`比`loadtxt()`更加强大，可对缺失数据进行处理。

【函数】

```python
def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n',header='', footer='', comments='# ', encoding=None):
	...
```

- `fname`：文件路径
- `X`：存入文件的数组。
- `fmt='%.18e'`：写入文件中每个元素的字符串格式，默认'%.18e'（保留18位小数的浮点数形式）。
- `delimiter=' '`：分割字符串，默认以空格分隔。

【函数】

```python
def loadtxt(fname, dtype=float, comments='#', delimiter=None,
            converters=None, skiprows=0, usecols=None, unpack=False,
            ndmin=0, encoding='bytes', max_rows=None):
    ...
```

- `fname`：文件路径。
- `dtype=float`：数据类型，默认为float。
- `comments='#'`: 字符串或字符串组成的列表，默认为'#'，表示注释字符集开始的标志。
- `skiprows=0`：跳过多少行，一般跳过第一行表头。
- `usecols=None`：元组（元组内数据为列的数值索引）， 用来指定要读取数据的列（第一列为0）。
- `unpack=False`：当加载多列数据时是否需要将数据列进行解耦赋值给不同的变量。

【例】写入和读出 TXT 文件。

```python
import numpy as np

outfile = r'.\test.txt'
x = np.arange(0, 10).reshape(2, -1)
np.savetxt(outfile, x)
y = np.loadtxt(outfile)
print(y)
# [[0. 1. 2. 3. 4.]
#  [5. 6. 7. 8. 9.]]
```

`test.txt` 文件如下：

```python
0.000000000000000000e+00 1.000000000000000000e+00 2.000000000000000000e+00 3.000000000000000000e+00 4.000000000000000000e+00
5.000000000000000000e+00 6.000000000000000000e+00 7.000000000000000000e+00 8.000000000000000000e+00 9.000000000000000000e+00
```

【例】写入和读出 CSV 文件。

```python
import numpy as np

outfile = r'.\test.csv'
x = np.arange(0, 10, 0.5).reshape(4, -1)
np.savetxt(outfile, x, fmt='%.3f', delimiter=',')
y = np.loadtxt(outfile, delimiter=',')
print(y)
# [[0.  0.5 1.  1.5 2. ]
#  [2.5 3.  3.5 4.  4.5]
#  [5.  5.5 6.  6.5 7. ]
#  [7.5 8.  8.5 9.  9.5]]
```

`test.csv` 文件如下：

```python
0.000,0.500,1.000,1.500,2.000
2.500,3.000,3.500,4.000,4.500
5.000,5.500,6.000,6.500,7.000
7.500,8.000,8.500,9.000,9.500
```

【函数】

```python
def genfromtxt(fname, dtype=float, comments='#', delimiter=None,
               skip_header=0, skip_footer=0, converters=None,
               missing_values=None, filling_values=None, usecols=None,
               names=None, excludelist=None,
               deletechars=''.join(sorted(NameValidator.defaultdeletechars)),
               replace_space='_', autostrip=False, case_sensitive=True,
               defaultfmt="f%i", unpack=None, usemask=False, loose=True,
               invalid_raise=True, max_rows=None, encoding='bytes'):
    ...
```

- `genfromtxt()`函数：从文本文件加载数据，并按指定方式处理缺少的值（是面向结构数组和缺失数据处理的。）。
- `names=None`：设置为True时，程序将把第一行作为列名称。

`data.csv` 文件（不带缺失值）

```python
id,value1,value2,value3
1,123,1.4,23
2,110,0.5,18
3,164,2.1,19
```

【例】

```python
import numpy as np

outfile = r'.\data.csv'
x = np.loadtxt(outfile, delimiter=',', skiprows=1)
print(x)
# [[  1.  123.    1.4  23. ]
#  [  2.  110.    0.5  18. ]
#  [  3.  164.    2.1  19. ]]

x = np.loadtxt(outfile, delimiter=',', skiprows=1, usecols=(1, 2))
print(x)
# [[123.    1.4]
#  [110.    0.5]
#  [164.    2.1]]

val1, val2 = np.loadtxt(outfile, delimiter=',', skiprows=1, usecols=(1, 2), unpack=True)
print(val1)  # [123. 110. 164.]
print(val2)  # [1.4 0.5 2.1]
```

【例】

```python
import numpy as np

outfile = r'.\data.csv'
x = np.genfromtxt(outfile, delimiter=',', names=True)
print(x)
# [(1., 123., 1.4, 23.) (2., 110., 0.5, 18.) (3., 164., 2.1, 19.)]

print(type(x))  
# <class 'numpy.ndarray'>

print(x.dtype)
# [('id', '<f8'), ('value1', '<f8'), ('value2', '<f8'), ('value3', '<f8')]

print(x['id'])  # [1. 2. 3.]
print(x['value1'])  # [123. 110. 164.]
print(x['value2'])  # [1.4 0.5 2.1]
print(x['value3'])  # [23. 18. 19.]
```

`data1.csv` 文件（带有缺失值）

```python
id,value1,value2,value3
1,123,1.4,23
2,110,,18
3,,2.1,19
```

【例】

```python
import numpy as np

outfile = r'.\data1.csv'
x = np.genfromtxt(outfile, delimiter=',', names=True)
print(x)
# [(1., 123., 1.4, 23.) (2., 110., nan, 18.) (3.,  nan, 2.1, 19.)]

print(type(x))  
# <class 'numpy.ndarray'>

print(x.dtype)
# [('id', '<f8'), ('value1', '<f8'), ('value2', '<f8'), ('value3', '<f8')]

print(x['id'])  # [1. 2. 3.]
print(x['value1'])  # [123. 110.  nan]
print(x['value2'])  # [1.4 nan 2.1]
print(x['value3'])  # [23. 18. 19.]
```

------

### 3. 文本格式选项

【函数】

```python
def set_printoptions(precision=None, threshold=None, edgeitems=None,
                     linewidth=None, suppress=None, nanstr=None, infstr=None,
                     formatter=None, sign=None, floatmode=None, **kwarg):
    ...
```

- `set_printoptions()`函数：设置打印选项。这些选项决定浮点数、数组和其它NumPy对象的显示方式。
- `precision=8`：设置浮点精度，控制输出的小数点个数，默认是8。
- `threshold=1000`：概略显示，超过该值则以“…”的形式来表示，默认是1000。
- `linewidth=75`：用于确定每行多少字符数后插入换行符，默认为75。
- `suppress=False`：当`suppress=True`，表示小数不需要以科学计数法的形式输出，默认是False。
- `nanstr=nan`：浮点非数字的字符串表示形式，默认`nan`。
- `infstr=inf`：浮点无穷大的字符串表示形式，默认`inf`。
- `formatter`：一个字典，自定义格式化用于显示的数组元素。键为需要格式化的类型，值为格式化的字符串。
  - `'bool'`
  - `'int'`
  - `'float'`
  - `'str'` : all other strings
  - `'all' `: sets all types
  - ...

【例】

```python
import numpy as np

np.set_printoptions(precision=4)
x = np.array([1.123456789])
print(x)  # [1.1235]

np.set_printoptions(threshold=20)
x = np.arange(50)
print(x)  # [ 0  1  2 ... 47 48 49]

np.set_printoptions(threshold=np.iinfo(np.int).max)
print(x)
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
#  48 49]

eps = np.finfo(float).eps
x = np.arange(4.)
x = x ** 2 - (x + eps) ** 2
print(x)  
# [-4.9304e-32 -4.4409e-16  0.0000e+00  0.0000e+00]
np.set_printoptions(suppress=True)
print(x)  # [-0. -0.  0.  0.]

x = np.linspace(0, 10, 10)
print(x)
# [ 0.      1.1111  2.2222  3.3333  4.4444  5.5556  6.6667  7.7778  8.8889
#  10.    ]
np.set_printoptions(precision=2, suppress=True, threshold=5)
print(x)  # [ 0.    1.11  2.22 ...  7.78  8.89 10.  ]

np.set_printoptions(formatter={'all': lambda x: 'int: ' + str(-x)})
x = np.arange(3)
print(x)  # [int: 0 int: -1 int: -2]

np.set_printoptions()  # formatter gets reset
print(x)  # [0 1 2]
```

【例】恢复默认选项

```python
np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75,
                    nanstr='nan', precision=8, suppress=False, 
                    threshold=1000, formatter=None)
```

【函数】

```python
def get_printoptions():
    ...
```

`get_printoptions()`函数：获取当前打印选项。

【例】

```python
import numpy as np

x = np.get_printoptions()
print(x)
# {
# 'edgeitems': 3, 
# 'threshold': 1000, 
# 'floatmode': 'maxprec', 
# 'precision': 8, 
# 'suppress': False, 
# 'linewidth': 75, 
# 'nanstr': 'nan', 
# 'infstr': 'inf', 
# 'sign': '-', 
# 'formatter': None, 
# 'legacy': False
# }
```

# 第七章 随机抽样

（略）

# 第八章 统计相关

## 一、次序统计

### 计算最小值

- `numpy.amin(a[, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue, where=np._NoValue])`：Return the minimum of an array or minimum along an axis.

【例】计算最小值

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.amin(x)
print(y)  # 11

y = np.amin(x, axis=0)
print(y)  # [11 12 13 14 15]

y = np.amin(x, axis=1)
print(y)  # [11 16 21 26 31]
```

### 计算最大值

- `numpy.amax(a[, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue, where=np._NoValue])`：Return the maximum of an array or maximum along an axis.

【例】计算最大值

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.amax(x)
print(y)  # 35

y = np.amax(x, axis=0)
print(y)  # [31 32 33 34 35]

y = np.amax(x, axis=1)
print(y)  # [15 20 25 30 35]
```

### 计算极差

- `numpy.ptp(a, axis=None, out=None, keepdims=np._NoValue)`： Range of values (maximum - minimum) along an axis. The name of the function comes from the acronym for 'peak to peak'.

【例】计算极差

```python
import numpy as np

np.random.seed(20200623)
x = np.random.randint(0, 20, size=[4, 5])
print(x)
# [[10  2  1  1 16]
#  [18 11 10 14 10]
#  [11  1  9 18  8]
#  [16  2  0 15 16]]

print(np.ptp(x))  # 18
print(np.ptp(x, axis=0))  # [ 8 10 10 17  8]
print(np.ptp(x, axis=1))  # [15  8 17 16]
```

### 计算分位数

- `numpy.percentile(a, q, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False)`： Compute the q-th percentile of the data along the specified axis. Returns the q-th percentile(s) of the array elements.
  - `a`：array，用来算分位数的对象，可以是多维的数组。
  - `q`：介于 0-100 的 float，用来计算是几分位的参数，如四分之一位就是 25，如要算两个位置的数就 [25, 75]。
  - `axis`：坐标轴的方向，一维的就不用考虑了，多维的就用这个调整计算的维度方向，取值范围 0/1。

【例】计算分位数

```python
import numpy as np

np.random.seed(20200623)
x = np.random.randint(0, 20, size=[4, 5])
print(x)
# [[10  2  1  1 16]
#  [18 11 10 14 10]
#  [11  1  9 18  8]
#  [16  2  0 15 16]]

print(np.percentile(x, [25, 50]))  
# [ 2. 10.]

print(np.percentile(x, [25, 50], axis=0))
# [[10.75  1.75  0.75 10.75  9.5 ]
#  [13.5   2.    5.   14.5  13.  ]]

print(np.percentile(x, [25, 50], axis=1))
# [[ 1. 10.  8.  2.]
#  [ 2. 11.  9. 15.]]
```

------

## 二、均值与方差

### 计算中位数

- `numpy.median(a, axis=None, out=None, overwrite_input=False, keepdims=False)` Compute the median along the specified axis. Returns the median of the array elements.

【例】计算中位数

```python
import numpy as np

np.random.seed(20200623)
x = np.random.randint(0, 20, size=[4, 5])
print(x)
# [[10  2  1  1 16]
#  [18 11 10 14 10]
#  [11  1  9 18  8]
#  [16  2  0 15 16]]
print(np.percentile(x, 50))
print(np.median(x))
# 10.0

print(np.percentile(x, 50, axis=0))
print(np.median(x, axis=0))
# [13.5  2.   5.  14.5 13. ]

print(np.percentile(x, 50, axis=1))
print(np.median(x, axis=1))
# [ 2. 11.  9. 15.]
```

### 计算平均值

- `numpy.mean(a[, axis=None, dtype=None, out=None, keepdims=np._NoValue)])`Compute the arithmetic mean along the specified axis.

【例】计算平均值（沿轴的元素的总和除以元素的数量）

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.mean(x)
print(y)  # 23.0

y = np.mean(x, axis=0)
print(y)  # [21. 22. 23. 24. 25.]

y = np.mean(x, axis=1)
print(y)  # [13. 18. 23. 28. 33.]
```

### 计算加权平均值

- `numpy.average(a[, axis=None, weights=None, returned=False])`Compute the weighted average along the specified axis.

【例】计算加权平均值（将各数值乘以相应的权数，然后加总求和得到总体值，再除以总的单位数）

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.average(x)
print(y)  # 23.0

y = np.average(x, axis=0)
print(y)  # [21. 22. 23. 24. 25.]

y = np.average(x, axis=1)
print(y)  # [13. 18. 23. 28. 33.]


y = np.arange(1, 26).reshape([5, 5])
print(y)
# [[ 1  2  3  4  5]
#  [ 6  7  8  9 10]
#  [11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]]

z = np.average(x, weights=y)
print(z)  # 27.0

z = np.average(x, axis=0, weights=y)
print(z)
# [25.54545455 26.16666667 26.84615385 27.57142857 28.33333333]

z = np.average(x, axis=1, weights=y)
print(z)
# [13.66666667 18.25       23.15384615 28.11111111 33.08695652]
```

### 计算方差

- `numpy.var(a[, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue])`Compute the variance along the specified axis.
  - `ddof=0`：是“Delta Degrees of Freedom”，表示自由度的个数。

要注意方差和样本方差的无偏估计，方差公式中分母上是 `n`；样本方差无偏估计公式中分母上是 `n-1`（`n` 为样本个数）。证明参见概率统计相关书籍。<br/>

【例】计算方差

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.var(x)
print(y)  # 52.0
y = np.mean((x - np.mean(x)) ** 2)
print(y)  # 52.0

y = np.var(x, ddof=1)
print(y)  # 54.166666666666664
y = np.sum((x - np.mean(x)) ** 2) / (x.size - 1)
print(y)  # 54.166666666666664

y = np.var(x, axis=0)
print(y)  # [50. 50. 50. 50. 50.]

y = np.var(x, axis=1)
print(y)  # [2. 2. 2. 2. 2.]
```

### 计算标准差

- `numpy.std(a[, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue])`Compute the standard deviation along the specified axis.

标准差是一组数据平均值分散程度的一种度量，是方差的算数平方根。<br/>

【例】计算标准差

```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.std(x)
print(y)  # 7.211102550927978
y = np.sqrt(np.var(x))
print(y)  # 7.211102550927978

y = np.std(x, axis=0)
print(y)
# [7.07106781 7.07106781 7.07106781 7.07106781 7.07106781]

y = np.std(x, axis=1)
print(y)
# [1.41421356 1.41421356 1.41421356 1.41421356 1.41421356]
```

------

## 三、相关性的衡量

### 计算协方差矩阵

- `numpy.cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None,aweights=None)` Estimate a covariance matrix, given data and weights.

【例】计算协方差矩阵

```python
import numpy as np

x = [1, 2, 3, 4, 6]
y = [0, 2, 5, 6, 7]
print(np.cov(x))  # 3.7   #样本方差
print(np.cov(y))  # 8.5   #样本方差
print(np.cov(x, y))
# [[3.7  5.25]
#  [5.25 8.5 ]]

print(np.var(x))  # 2.96    #方差
print(np.var(x, ddof=1))  # 3.7    #样本方差
print(np.var(y))  # 6.8    #方差
print(np.var(y, ddof=1))  # 8.5    #样本方差

z = np.mean((x - np.mean(x)) * (y - np.mean(y)))    #协方差
print(z)  # 4.2

z = np.sum((x - np.mean(x)) * (y - np.mean(y))) / (len(x) - 1)   #样本协方差
print(z)  # 5.25

z = np.dot(x - np.mean(x), y - np.mean(y)) / (len(x) - 1)     #样本协方差     
print(z)  # 5.25
```

### 计算相关系数

- `numpy.corrcoef(x, y=None, rowvar=True, bias=np._NoValue, ddof=np._NoValue)` Return Pearson product-moment correlation coefficients.

理解了 `np.cov()` 函数之后，很容易理解 `np.correlate()` 函数，二者的参数几乎一样。<br/>

`np.cov()` 函数描述的是两个向量协同变化的程度，它的取值可能非常大，也可能非常小，这就导致没办法直观地衡量二者协同变化的程度。相关系数实际上是正则化的协方差，`n` 个变量的相关系数形成一个 `n` 维方针。<br/>

【例】计算相关系数

```python
import numpy as np

np.random.seed(20200623)
x, y = np.random.randint(0, 20, size=(2, 4))

print(x)  # [10  2  1  1]
print(y)  # [16 18 11 10]

z = np.corrcoef(x, y)
print(z)
# [[1.         0.48510096]
#  [0.48510096 1.        ]]

a = np.dot(x - np.mean(x), y - np.mean(y))
b = np.sqrt(np.dot(x - np.mean(x), x - np.mean(x)))
c = np.sqrt(np.dot(y - np.mean(y), y - np.mean(y)))
print(a / (b * c))  # 0.4851009629263671
```

------

## 四、其它

### 直方图

- `numpy.digitize(x, bins, right=False)`Return the indices of the bins to which each value in input array belongs.
  - `x`：numpy 数组
  - `bins`：一维单调数组，必须是升序或者降序
  - `right`：间隔是否包含最右
  - 返回值：x 在 bins 中的位置。

【例】

```python
import numpy as np

x = np.array([0.2, 6.4, 3.0, 1.6])
bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
inds = np.digitize(x, bins)
print(inds)  # [1 4 3 2]
for n in range(x.size):
    print(bins[inds[n] - 1], "<=", x[n], "<", bins[inds[n]])

# 0.0 <= 0.2 < 1.0
# 4.0 <= 6.4 < 10.0
# 2.5 <= 3.0 < 4.0
# 1.0 <= 1.6 < 2.5
```

【例】

```python
import numpy as np

x = np.array([1.2, 10.0, 12.4, 15.5, 20.])
bins = np.array([0, 5, 10, 15, 20])
inds = np.digitize(x, bins, right=True)
print(inds)  # [1 2 3 4 4]

inds = np.digitize(x, bins, right=False)
print(inds)  # [1 3 3 4 5]
```

# 第九章 线性代数

## 一、线性代数基础

Numpy 定义了 `matrix` 类型，使用该 `matrix` 类型创建的是矩阵对象，它们的加减乘除运算缺省采用矩阵方式计算，因此用法和Matlab 十分类似。但是由于 NumPy 中同时存在 `ndarray` 和 `matrix` 对象，因此用户很容易将两者弄混。这有违 Python 的“显式优于隐式”的原则，因此官方并不推荐在程序中使用 `matrix`。在这里，我们仍然用 `ndarray` 来介绍。

### 矩阵和向量积

矩阵的定义、矩阵的加法、矩阵的数乘、矩阵的转置与二维数组完全一致，不再进行说明，但矩阵的乘法有不同的表示。

- `numpy.dot(a, b[, out])`：计算两个矩阵的乘积，如果是一维数组则是它们的内积。

【例1】

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])
z = np.dot(x, y)
print(z)  # 70

x = np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
print(x)
# [[1 2 3]
#  [3 4 5]
#  [6 7 8]]

y = np.array([[5, 4, 2], [1, 7, 9], [0, 4, 5]])
print(y)
# [[5 4 2]
#  [1 7 9]
#  [0 4 5]]

z = np.dot(x, y)
print(z)
# [[  7  30  35]
#  [ 19  60  67]
#  [ 37 105 115]]

z = np.dot(y, x)
print(z)
# [[ 29  40  51]
#  [ 76  93 110]
#  [ 42  51  60]]
```

【注】：线代中的维数和数组的维数不同。如线代中提到的 n 维行向量在 Numpy 中是一维数组，而线代中的 n 维列向量在 Numpy 中是一个 shape 为 (n, 1) 的二维数组。

------

### 矩阵特征值与特征向量

- `numpy.linalg.eig(a)` ：计算方阵的特征值和特征向量。
- `numpy.linalg.eigvals(a)` ：计算方阵的特征值。

【例1】求方阵的特征值和特征向量

```python
import numpy as np

# 创建一个对角矩阵
x = np.diag((1, 2, 3))  
print(x)
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]

print(np.linalg.eigvals(x))
# [1. 2. 3.]

a, b = np.linalg.eig(x)  
# 特征值保存在a中，特征向量保存在b中
print(a)
# [1. 2. 3.]
print(b)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# 检验特征值与特征向量是否正确
for i in range(3): 
    if np.allclose(a[i] * b[:, i], np.dot(x, b[:, i])):
        print('Right')
    else:
        print('Error')
# Right
# Right
# Right
```

【例2】判断对称矩阵是否为正定阵（特征值是否全部为正）。

```python
import numpy as np

A = np.arange(16).reshape(4, 4)
print(A)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
#  [12 13 14 15]]

A = A + A.T  # 将方阵转换成对称阵
print(A)
# [[ 0  5 10 15]
#  [ 5 10 15 20]
#  [10 15 20 25]
#  [15 20 25 30]]

B = np.linalg.eigvals(A)  # 求A的特征值
print(B)
# [ 6.74165739e+01 -7.41657387e+00  1.82694656e-15 -1.72637110e-15]

# 判断是不是所有的特征值都大于0，用到了all函数，显然对称阵A不是正定的
if np.all(B > 0):
    print('Yes')
else:
    print('No')
# No
```

## 二、矩阵分解

### 奇异值分解


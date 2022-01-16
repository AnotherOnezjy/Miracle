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


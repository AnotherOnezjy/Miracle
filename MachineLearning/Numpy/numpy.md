# 一、常量

## numpy.nan

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

##  numpy.inf

- 表示正无穷大

  Inf = inf = infty = Infinity = PINF

  NINF = -inf

【例】

```python
import numpy as np

print(np.inf) # inf
print(np.NINF) # -inf
```

## numpy.pi

- 表示圆周率

【例】

```python
import numpy as np

print(np.pi)  # 3.141592653589793
```

## numpy.e

- 表示自然常数

【例】

```python
import numpy as np

print(np.e)  # 2.718281828459045
```

# 二、数据类型

## 常见数据类型

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

## 创建数据类型

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

## 数据类型信息

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

# 三、时间日期和时间增量

## datetime64基础

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

## datetime64 和 timedelta64 运算

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

## datetime64 的应用

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

# 四、数组的创建

numpy 提供的最重要的数据结构是 `ndarray`，它是 python 中 `list` 的扩展。

## 1. 依据现有数据来创建 ndarray

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

## 2. 依据 ones 和 zeros 填充方式

在机器学习任务中经常做的一件事就是初始化常数，需要用常数值或随机值来创建一个固定大小的矩阵。

### (a) 零数组

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

### (b) 1 数组

- `ones()` 函数：返回给定形状和类型的 1 数组；
- `ones_like()` 函数：返回与给定数组形状和类型相同的 1 数组。

```python
def ones(shape, dtype=None, order='C'):
    ...
    
def ones_like(a, dtype=None, order='K', subok=True, shape=None):
    ...
```

### (c) 空数组

- `empty()` 函数：返回一个空数组，数组元素为随机数；
- `empty_like` 函数：返回与给定数组具有相同形状和类型的新数组。

```python
def empty(shape, dtype=None, order='C'):
    ...
    
def empty_like(prototype, dtype=None, order='K', subok=None, shape=None):
    ...
```

### (d) 单位数组

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

### (e) 对角数组

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

### (f) 常数数组

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

## 3. 利用数值范围来创建 ndarray

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

## 4. 结构数组的创建

结构数组，首先需要定义结构，然后利用 `np.array()` 来创建数组，其参数 `dtype` 为定义的结构。

### (a) 利用字典来定义结构

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

### (b) 利用包含多个元组的列表来定义结构

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


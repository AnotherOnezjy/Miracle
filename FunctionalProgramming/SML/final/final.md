# 12.21.2021 周二 随堂练习

### 1

从功能和性能两方面比较下列两个函数的异同点。

```sml
fun take( [ ], i) = [ ]
    | take(x::xs, i) = if i > 0 then x::take(xs, i-1)
 			   else [ ];
```

```sml
fun rtake ([ ], _, taken) = taken
     | rtake (x::xs,i,taken) =  if i>0 then rtake(xs, i-1, x::taken)
		  		   else taken;
```

-   功能

1. 函数 `take` 作用是取表的前 i 个元素
2. 函数 `rtake` 作用是取表的前 i 个元素**倒序**接在 `taken` 前

-   性能

1. 函数 `take` 复杂度是 $O(n)$ 的
2. 函数 `rtake` 复杂度是 $O(n)$ 的

### 2

编写函数 `evens: int list -> int list`，要求对所有 `int list L`, `evens(L)` 执行的结果为 L 中所有偶数的子集，且该子集中数据的顺序与`L`中出现的顺序一致。
例如：
`evens [1, 2, 3, 4] = [2, 4]`
`evens [1, 3, 5, 7] = [ ]`

```sml
fun isOdd(0) = true
    | isOdd(1) = false
    | isOdd(x) = isOdd(x - 2);

fun even [] = []
    | even (x::L) =
        if isOdd(x) then x::even(L)
        else even(L);

val test1 = [2, 4, 6, 7, 8];
val test2 = [1, 3, 5, 7];
val test3 = [1, 3, 5, 8];

val res1 = even(test1);
val res2 = even(test2);
val res3 = even(test3);
```

### 3

1. 编写高阶函数：`toInt: int -> int list -> int`. 对所有 `b>1` 和所有 `L: int list`,如果 `L` 是一个 `b` 进制数的 `int list` 表示，函数 `toInt b L` 为其相应的整数值，`toInt b` 的结果类型为：`int list -> int`.

如：
`val base2ToInt = toInt 2;`
`val 2 = base2ToInt [0,1];`

提示：

```sml
fun toInt (base : int) (digits : int list) : int =
    case digits of
      [] =>  _________
    | d::digits’ =>  _________
```

-   code

```sml
(* toInt: int -> int list -> int *)
(* REQUIRE: L为b（b>1）进制数的int list表示 *)
(* ENSURE: toInt b L为其相应的整数值 *)
fun toInt(base: int) (digits: int list): int  =
    case digits of
        [] => 0
        | d::digits => d + (base * (toInt base digits))

(* 测试 *)
val base2ToInt = toInt 2;
base2ToInt [0,1];       (* 结果应为 2 *)
base2ToInt [1,0,1];     (* 结果应为 5 *)
```

2. 利用数学操作 `mod` 和 `div` 可以将任意十进制整数 `n` 表示成基于基数 `b` 的 `b` 进制数形式，如 `4210=1325`。
   编写高阶函数 `toBase: int -> int -> int list` 实现该转换：`toBase b n`将十进制数 `n` 转换为 `b` 进制数的`int list`表述形式`（b>1, n≥0）`。

-   code

```sml
(* toBase: int -> int -> int list *)
(* REQUIRE: n为10进制，且n>=0,b>1 *)
(* ENSURE: toBase b n 将十进制数n转换为b进制数的int list形式 *)
fun toBase (base: int) 0 = []
    | toBase (base: int) (n: int) =
        let
            val (q, r) = (n div base, n mod base)
        in
            r::(toBase base q)
        end;

(* 测试 *)
toBase 5 42;        (* 结果应为[2,3,1] *)
```

1. 编写高阶函数 `convert: int * int -> int list -> int list`
   对任意`b1, b2 > 1`和所有`L: int list（L为一个b1进制数的int list表述形式）`，函数`convert(b1, b2)` L 将 b1 进制数的`int list`表述 L 转换成 b2 进制数的`int list`表述，即满足 `toInt b2 (convert(b1, b2) L) = toInt b1 L`。

-   code

```sml
(* convert: int * int -> int list -> int list *)
(* REQUIRE: b1,b2>1，L为b1进制数的int list表示 *)
(* ENSURE: convert(b1,b2) L 将b1进制数的列表形式
 * 转换为b2进制数列表形式
 * 即满足 toInt b2(convert(b1,b2) L) = toInt b1 L
 *)
fun convert (b1, b2) L =
        let
            val n = toInt b1 L
        in
            toBase b2 n
        end;

(* 测试 *)
toInt 5 (convert(10,5) [2,4]) = toInt 10 [2,4];
```

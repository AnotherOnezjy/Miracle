## 实验一 11.30

### 1.下列模式能否与类型为 `int list` 的 L 匹配成功？如果匹配不成功，指出该模式的类型？（假设 x 为 int 类型）

1.  `x::L`
    匹配成功
2.  `_::_`
    匹配成功
3.  `x::(y::L)`
    当 `list` 成员数量大于等于 2 时，匹配成功
4.  `(x::y)::L`
    失败，匹配的是 `int list list`
5.  `[x, y]`
    当 `list` 成员数量为 2 时，匹配成功

### 2. 试写出与下列表述相对应的模式。如果没有模式与其对应，试说明原因。

1.  list of length 3
    `[x, y, z]`
2.  lists of length 2 or 3
    无，可以拆分为 `[x, y]` 和 `[x, y, z]` 两个模式分别进行匹配
3.  Non-empty lists of pairs
    `(x, y)::L`
4.  Pairs with both components being non-empty lists
    `(x::L1, y::L2)`

### 3. 分析下述程序段（左边括号内为标注的行号）：

<img src=".\T3code.png"></img>

试问：第 4 行中的 x、第 5 行中的 m 和第 6 行中的 x 的声明绑定的类型和值分别为什么？第 14 行表达式 assemble(x, 3.0)计算的结果是什么？

-   第 4 行中 x : int, 2
-   第 5 行中 m : real, 12.4
-   第 6 行中 x : int, 9001
-   第 14 行计算结果：27

### 4. 指出下列代码的错误：

```sml
(* pi: real *)
val pi : real = 3.14159;

(* fact: int -> int *)
fun fact (0 : int) : int = 1
  | fact n = n * (fact (n - 1));

(* f : int -> int *)
fun f (3 : int) : int = 9
    f _ = 4;                                (*Error 1: 此行开头应加入一个 '|' 字符*)

(* circ : real -> real *)
fun circ (r : real) : real = 2 * pi * r     (*Error 2: 2 为 int 类型，应该改为 real 类型的 2.0*)

(* semicirc : real -> real *)
fun semicirc : real = pie * r
(*Error 3
    函数 semicirc 没有匹配参数 r，而且将 pi 错拼为 pie，正确的写法为：
    fun semicirc : (r : real) : real = pi * r;
*)

(* area : real -> real *)
fun area (r : int) : real = pi * r * r      (*Error 4: 参数 r 类型错误，应为 real 类型*)

(*Error 5: 除此之外，函数 circ、semicirc 以及 area 的末尾都少了一个分号 ';'*)
```

### 5. 在提示符下依次输入下列语句，观察并分析每次语句的执行结果。

-   `3 + 4;`
-   `3 + 2.0;`
-   `it + 6;`
-   `val it = “hello”;`
-   `it + “ world”;`
-   `it + 5;`
-   `val a = 5;`
-   `a = 6;`
-   `a + 8;`
-   `val twice = (fn x => 2 * x);`
-   `twice a;`
-   `let x = 1 in x end;`
-   `foo;`
-   `[1, “foo”];`

<img src=".\T5-1.png"></img>
<img src=".\T5-2.png"></img>

更正后的语句如下：

```sml
(*以下是更改后正确的代码*)
3 + 4;
3.0 + 2.0;              (*类型不一样*)
it + 6.0;               (*类型不一样*)
val it = "hello";
it ^ "world";           (*需要用 ^ 连接*)
(*it + 5;(*类型不匹配*)*)
val a = 5;
val a = 6;
a + 8;
val twice = (fn x => 2 * x);
twice a;
let val x = 1 in x end; (*需要在 x 前面加上 val*)
(*foo;无意义的语句*)
(*[1,"foo"]; int 类型和 string 类型不能放在一个 lists 里面*)
```

### 6. 函数 `sum` 用于求解整数列表中所有整数的和，函数定义如下：

```sml
(* sum : int list -> int 		*)
(* REQUIRES: true		*)
(* ENSURES: sum(L) evaluates to the sum of the integers in L. *)
fun sum [ ] = 0
    | sum (x::L) = x + (sum L);
```

完成函数 mult 的编写，实现求解整数列表中所有整数的乘积。

```sml
(* mult : int list -> int 		*)
(* REQUIRES: true		*)
(* ENSURES: mult(L) evaluates to the product of the integers in L. *)
fun mult [ ] = 		(* FILL IN *)
    | mult (x::L) = 	(* FILL IN *) 
```

完整程序及测试结果截图如下：

-   code

```sml
(* sum : int list -> int 		*)
(* REQUIRES: true		*)
(* ENSURES: sum(L) evaluates to the sum of the integers in L. *)
fun sum [] = 0
    | sum (x::L) = x + (sum L);

val a = [1, 2, 3, 4];
val b = sum(a);

(* mult : int list -> int 		*)
(* REQUIRES: true		*)
(* ENSURES: mult(L) evaluates to the product of the integers in L. *)
fun mult [] = 0
    | mult(x::L) = if L = [] then x else x * (mult L);

val c = [1, 2, 3, 4];
val d = mult(c);
```

-   pic
    <img src=".\T6.png"></img>

### 7. 编写函数实现下列功能：

1. `zip: string list * int list -> (string * int) list`
   其功能是提取第一个 string list 中的第 i 个元素和第二个 int list 中的第 i 个元素组成结果 list 中的第 i 个二元组。如果两个 list 的长度不同，则结果的长度为两个参数 list 长度的最小值。
2. `unzip: (string * int) list -> string list * int list`
   其功能是执行 zip 函数的反向操作，将二元组 list 中的元素分解成两个 list，第一个 list 中的元素为参数中二元组的第一个元素的 list，第二个 list 中的元素为参数中二元组的第二个元素的 list。

对所有元素 L1: string list 和 L2: int list，unzip( zip (L1, L2)) = (L1, L2)是否成立？如果成立，试证明之；否则说明原因。

-   code

```sml
fun zip([], _) = []
    | zip(_, []) = []
    | zip(s::(SL: string list), x::(IL: int list)) = (s, x)::zip(SL, IL);

fun unzip([]) = ([], [])
    | unzip((s: string, x: int)::L) =
        let
            val (SL, IL) = unzip(L)
        in
            (s::SL, x::IL)
        end

val s1: string list = ["a", "b", "c"];
val s2: int list = [1, 2, 3];
val s3: int list = [1, 2];
val s4: int list = [1, 2, 3, 4, 5];
zip(s1, s2);(*模式正确且长度相等*)
zip(s1, s3);(*模式正确，前长后短*)
zip(s1, s4);(*模式正确，前短后长*)
(* zip(s2, s1); *)(*模式错误*)

val ss : (string * int) list = [("a", 1), ("b", 2)];
unzip(ss);
```

不一定成立，因为 L1 和 L2 的长度可能不同。若 L1 = ["a", "b", "c"]，L2 = [1, 2]，则 unzip(zip(L1, L2)) 的结果为 (["a", "b"], [1, 2])，可以看出 ["a", "b"] 和原来的 L1 不等。

### 8. 完成如下函数 `Mult: int list list -> int` 的编写,该函数调用 `mult` 实现 `int list list` 中所有整数乘积的求解。

```sml
(* mult : int list list -> int 	*)
(* REQUIRES: true		*)
(* ENSURES: mult(R) evaluates to the product of all the integers in the lists of R. *)
 
fun Mult [ ] = 	(* FILL IN *)
    | Mult (r :: R) = 	(* FILL IN *)
```

-   code

```sml
val a = [[1, 2], [3, 4]];

fun mult [] = 0
    | mult(x::L) = if L = [] then x else x * (mult L);

fun Mult [] = 0
    | Mult(x::L) = if L = [] then mult(x) else mult(x) * (Mult L);

val b = Mult(a);
```

### 9. 函数 `mult’`定义如下，试补充其函数说明，指出该函数的功能。

```sml
(* mult’ : int list * int -> int 			*)
(* REQUIRES: true				*)
(* ENSURES: mult’(L, a) … (* FILL IN *) 	*)

 fun mult’ ([ ], a) = a
	  | mult’ (x :: L, a) = mult’ (L, x * a);
```

利用 `mult’`定义函数 `Mult’ : int list list * int -> int`，使对任意整数列表的列表 R 和整数 a，该函数用于计算 a 与列表 R 中所有整数的乘积。该函数框架如下所示，试完成代码的编写。

```sml
fun Mult’ ( [ ], a) = 	(* FILL IN *)
    | Mult’ (r::R, a) = 	(* FILL IN *)
```

-   code

```sml
(* mult' : int list * int -> int  *)
(* REQUIRES: true *)
(* ENSURES: mult'(L, a) a 与列表 L 中元素的乘积 *)
fun mult' ([], a) = a
    | mult' (x::L, a) = mult'(L, x * a);

(* mult' : int list list * int -> int  *)
(* REQUIRES: true *)
(* ENSURES: Mult'(R, a) a 与列表 R 中所有整数的乘积 *)
fun Mult'([], a) = a
    | Mult'(r::R, a) = mult'(r, a) * Mult'(R, 1);

mult'([1, 2, 3, 4], 2);
Mult'([[1, 2, 3], [4, 5, 6]], 2);
```

### 10. 编写递归函数 `square` 实现整数平方的计算，即 `square n = n * n`。

要求：程序中可调用函数 double，但不能使用整数乘法（\*）运算。

```sml
(* double : int -> int *)
(* REQUIRES: n >= 0 *)
(* ENSURES: double n evaluates to 2 * n.*)

fun double (0 : int) : int = 0
    | double n = 2 + double (n - 1)
```

分析：$2n = 2(n-1) + 2$

-   code

```sml
(* double : int -> int *)
(* REQUIRES: n >= 0 *)
(* ENSURES: double n evaluates to 2 * n.*)

fun double (0 : int) : int = 0
    | double n = 2 + double (n - 1)

(* square : int -> int *)
(* REQUIRES: n >= 0 *)
(* ENSURES: square n evaluates to n * n.*)
fun square(0 : int) = 0
    | square n = square(n - 1) + double(n) - 1;

square(5);
square(100);
```

思路：$n^2 = (n-1)^2 + 2n - 1$

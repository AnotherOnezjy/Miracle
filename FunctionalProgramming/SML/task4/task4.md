## 实验四

### 1

函数 map 和 filter 定义如下：

-   `map`

```sml
(* map: ('a -> 'b) -> 'a list -> 'b list *)
fun map f [ ] = [ ]
    | map f (x::xs) = (f x) :: (map f xs)
```

-   `filter`

```sml
(* filter: ('a -> bool) -> 'a list -> 'a list *)
fun filter f [] = []
    | filter f (x::xs) = if f x then x :: (filter f xs)
			     else filter f xs
```

推导下列表达式的类型和计算结果，并描述其功能。

```sml
map (filter (fn a => size a = 4)) [[“Sunday”, “Monday”], [“one”, “two”, “three”, “four”, “five”], [“year”, “month”, “day”]]
```

### 2

编写函数 `thenAddOne`，要求：

1. 函数类型为：`((int -> int) * int) -> int`
2. 功能为将一个整数通过函数变换（如翻倍、求平方或求阶乘）后再加 1

-   code

```sml
(* thenAddOne: ((int->int) * int) -> int *)
(* REQUIRES: *)
(* input parameter1 is a function with type mapping from int to int *)
(* input parameter2 is an int parameter *)
(* ENSURES: *)
(* the function returns the result that is bigger than the result performed by parameter1 on parameter2 by 1 *)
fun thenAddOne(func:(int -> int), x: int) = func(x) + 1;

(* squareCal: int -> int *)
(* REQUIRES: a number *)
(* ENSURES: return x * x *)
fun squareCal(x: int) = x * x;

(* doubleCal: int -> int *)
(* REQUIRES: a number *)
(* ENSURES: return 2x *)
fun doubleCal(x: int) = x + x;

(* factorial: int -> int *)
(* REQUIRES: a number *)
(* ENSURES: return x! *)
fun factorial(0) = 1
    | factorial(1) = 1
    | factorial(x: int) = x * factorial(x - 1);

(* a few tests *)
thenAddOne(doubleCal, 5); (*Out: 10*)
thenAddOne(squareCal, 5); (*Out: 26*)
thenAddOne(factorial, 5); (*Out: 126*)
```

### 3

编写函数 `mapList`，要求：

1. 函数类型为：`(('a -> 'b) * 'a list) -> 'b list`
2. 功能为实现整数集的数学变换（如翻倍、求平方或求阶乘）

-   code

```sml
(* mapList : (('a->'b) * 'a list) -> 'b list *)
(* REQUIRES: *)
(* input parameter1 is a function with polymorphic type mapping from 'a to 'b *)
(* input parameter2 is a polymorphic list 'a list *)
(* ENSURES: *)
(* the function calculates every result performed with elements in 'a list by parameter1 *)
(* and combines them as a list *)
fun mapList(func: ('a -> 'b), []: 'a list): 'b list = []
    | mapList(func, x::L) = func(x)::mapList(func, L);

fun squareCal(x: int) = x * x;

fun doubleCal(x: int) = x + x;

fun factorial(0) = 1
    | factorial(1) = 1
    | factorial(x: int) = x * factorial(x - 1);

val a = [~1, 1, 2, 3, 4, 5];
val b = [1, 2, 3, 4, 5];

(* a few tests *)
mapList(squareCal, a); (*Out: [1,1,4,9,16,25]*)
mapList(doubleCal, a); (*Out: [~2,2,4,6,8,10]*)
mapList(factorial, b); (*Out: [1,2,6,24,120]*)
```

### 4

编写函数 `mapList'`，要求：

1. 函数类型为：`('a -> 'b) -> ('a list -> 'b list)`
2. 功能为实现整数集的数学变换（如翻倍、求平方或求阶乘）
3. 比较函数 `mapList'` 和 `mapList`，分析、体会它们有什么不同

-   code

```sml
(* mapList : (('a->'b) -> ('a list ->'b list) *)
(* REQUIRES: *)
(* input parameter1 is a function with polymorphic type mapping from 'a to 'b *)
(* ENSURES: *)
(* output function with polymorphic type mapping from 'a list to 'b list *)
fun mapList'(func:('a -> 'b)): ('a list -> 'b list) =
        let
            fun res [] = []
                | res (x::L) = func(x)::res(L)
        in
            res
        end;

fun squareCal(x: int) = x * x;
fun doubleCal(x: int) = x + x;
fun factorial(0) = 1
    | factorial(1) = 1
    | factorial(x: int) = x * factorial(x - 1);

val a = [~1,1,2,3,4,5];
val b = [1,2,3,4,5];

(* a few tests *)
mapList'(squareCal)(a); (*Out: [1,1,4,9,16,25]*)
mapList'(doubleCal)(a); (*Out: [~2,2,4,6,8,10]*)
mapList'(factorial)(b); (*Out: [1,2,6,24,120]*)
```

### 5

编写函数：
`exists:('a -> bool) -> 'a list -> bool`
`forall:('a -> bool) -> 'a list -> bool`
对函数 `p: t -> bool`，整数集 `L: t list`,
有：
`exist p L =>_ true if there is an x in L such that p x=true;`
`exist p L =>_ false otherwise.`
`forall p L =>_ true if p x = true for every item x in L;`
`forall p L =>_ false otherwise.`

-   code

```sml
(* exists : (‘a -> bool) -> ‘a list -> bool *)
(* REQUIRES: *)
(* input is a function that gets 'a type and return bool type *)
(* ENSURES: *)
(* true if there is an x in L such that p x=true *)
(* false otherwise. *)
fun exists(func:('a -> bool)): ('a list -> bool) =
        let
            fun help [] = false
                | help (x::L) =
                    if func(x) then true
                    else help(L)
        in
            help
        end;

fun isOdd(0) = true
    | isOdd(1) = false
    | isOdd(x) = isOdd(x - 2);

val test1 = [2, 4, 6, 8, 10];
exists isOdd test1; (*Out: true*)

(* forall : (‘a -> bool) -> ‘a list -> bool *)
(* REQUIRES: *)
(* input is a function that gets 'a type and return bool type *)
(* ENSURES: *)
(* true if p x = true for every item x in L*)
(* false otherwise.*)
fun forall(func:('a -> bool)): ('a list -> bool) =
        let
            fun help [] = true
                | help (x::L) =
                    if (func(x)) then help(L)
                    else false
        in
            help
        end;

val test2 = [1, 3, 5, 7, 9];
forall isOdd test2; (*Out: false*)
forall isOdd test1; (*Out: true*)
```

### 6

编写函数：
`treeFilter: (‘a -> bool) -> ‘a tree -> ‘a option tree`
将树中满足条件 P（ ‘a -> bool ）的节点封装成 option 类型保留，否则替换成 NONE。

-   code

```sml
(*datatype 'a option = NONE | SOME of 'a;*)
datatype 'a tree = Empty | Node of 'a tree * 'a * 'a tree;

(* treeFilter : (‘a -> bool) -> ‘a tree -> ‘a option tree *)
(* REQUIRES: *)
(* input is a function that gets 'a type and return bool type*)
(* ENSURES: *)
(* capsule every node as type option that satisfies parameter1 in the tree *)
(* else, replace them as NONE *)
fun treeFilter(P: 'a -> bool): ('a tree->'a option tree) =
        let
            fun func(Empty) = Empty
                | func(x: 'a tree)=
                    let
                        val Node(l, v, r) = x;
                        val newv =
                            if P(v) then SOME v
                            else NONE
                    in
                        Node(func(l), newv, func(r))
                    end;
        in
            func
        end;

fun isOdd(x: int)=
        if x mod 2 = 1 then true
        else false;

fun split L =
        let
            val left = List.take(L, length(L) div 2)
            val right = List.drop(L, length(L) div 2)
        in 
            if length(right) = 0 then (left, 0, right)
            else (left, hd(right), List.drop(right, 1))
        end;

fun listToTree [] = Empty
    | listToTree (x::[]) = Node(Empty, x, Empty)
    | listToTree (x::L) =
        let 
            val (L1, y, L2) = split(x::L)
        in 
            Node(listToTree(L1),y,listToTree(L2))
        end;

fun trav (Empty)=[]
    | trav T =
        let 
            val Node(l, v, r) = T
        in 
            trav(l) @ (v::trav(r))
        end;

val L1 = [1, 2, 3, 4, 5];
val L2 = [~1, ~2, ~3, ~4, ~5, ~6, ~7];
val L3 = [1, 3, 5, 7, 9, 11, 13];

treeFilter isOdd (listToTree(L1));
treeFilter isOdd (listToTree(L2));
treeFilter isOdd (listToTree(L3));

trav (treeFilter isOdd (listToTree(L1)));
trav (treeFilter isOdd (listToTree(L2)));
trav (treeFilter isOdd (listToTree(L3)));
```

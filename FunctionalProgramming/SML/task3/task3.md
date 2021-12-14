## 实验三

### 0

预置代码

```sml
(*树类型定义*)
datatype tree = Empty | Node of tree * int * tree;

(*树的中序遍历*)
fun trav Empty = []
    | trav(Node(t1, x, t2)) = trav t1 @ (x::trav t2);
```

### 1

编写函数 `listToTree: int list -> tree`，将一个表转换成一棵平衡树。

提示：可调用 `split` 函数，`split` 函数定义如下：

如果 L 非空，则存在 L1, x, L2，满足：

split L = (L1, x, L2) 且
L = L1 @ x::L2 且
length(L1) 和 length(L2) 差值小于 1

-   code1

```sml
(*树类型定义*)
datatype tree = Empty | Node of tree * int * tree;

(*树的中序遍历*)
fun trav Empty = []
    | trav(Node(t1, x, t2)) = trav t1 @ (x::trav t2);

(*split: int list -> int list * int list*)
(*将 list 拆分为长度相差小于 1 的两个 list*)
fun split [] = ([], 0, [])
    | split [x] = ([], x, [])
    | split (x::L) =
        let
            val(A, y, B) = split L
        in
            if length(A) > length(B) then
                (A, x, y::B)
            else
                (y::A, x, B)
        end;

(*listToTree: int list -> tree*)
(*将一个表转为平衡树*)
fun listToTree [] = Empty
    | listToTree [x] = Node(Empty, x, Empty)
    | listToTree L =
        let
            val(A, y, B) = split L
        in
            Node(listToTree A, y, listToTree B)
        end;

(*测试*)
val l1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
val tree1 = listToTree l1;
trav tree1;
```

-   code2

```sml
(*树类型定义*)
datatype tree = Empty | Node of tree * int * tree;

(*树的中序遍历*)
fun trav Empty = []
    | trav(Node(t1, x, t2)) = trav t1 @ (x::trav t2);

(*split: int list -> int list * int list*)
(*将 list 拆分为长度相差小于 1 的两个 list*)
fun split [] = ([], 0, [])
    | split [x] = ([], x, [])
    | split (x::L) =
        let
            val(A, y, B) = split L
        in
            if length(A) > length(B) then
                (A, x, y::B)
            else
                (y::A, x, B)
        end;

fun listToTree [] = Empty
    | listToTree (x::L) =
        let
            val index = List.length L div 2
            val ltree = List.take(L, index)
            val rtree = List.drop(L, index)
        in
            Node(listToTree(ltree), x, listToTree(rtree))
        end;

(*测试*)
val l1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
val l2 = [100, 200, 300];
val tree1 = listToTree l1;
val tree2 = listToTree l2;
trav tree1;
trav tree2;
```

### 2

编写函数 `revT: tree -> tree`，对树进行反转，使 `trav(revT t) = reverse(trav t)`。（`trav` 为树的中序遍历函数）。假设输入参数为一棵平衡二叉树，验证程序的正确性，并分析该函数的执行性能（work 和 span）。

-   code

```sml
(*树类型定义*)
datatype tree = Empty | Node of tree * int * tree;

(*树的中序遍历*)
fun trav Empty = []
    | trav(Node(t1, x, t2)) = trav t1 @ (x::trav t2);

(*split: int list -> int list * int list*)
(*将 list 拆分为长度相差小于 1 的两个 list*)
fun split [] = ([], 0, [])
    | split [x] = ([], x, [])
    | split (x::L) =
        let
            val(A, y, B) = split L
        in
            if length(A) > length(B) then
                (A, x, y::B)
            else
                (y::A, x, B)
        end;

fun listToTree [] = Empty
    | listToTree (x::L) =
        let
            val index = List.length L div 2
            val ltree = List.take(L, index)
            val rtree = List.drop(L, index)
        in
            Node(listToTree(ltree), x, listToTree(rtree))
        end;

fun revT(T) = case T of Empty => Empty
        | Node(ltree, x, rtree) => Node(revT(rtree), x, revT(ltree));

(*测试*)
val l1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
val t1 = listToTree(l1);
trav t1;
val test = revT(t1);
trav test;
```

-   性能分析

因为每一层交换只需要一个步骤,而树的深度为 n 所以在并行计算时其执行性能为: $span = O(log_2 n)$ $work = O(n)$

### 3

编写函数 `binarySearch: tree * int -> bool`。当输出参数 1 为有序树时，如果树中包含值为参数 2 的节点，则返回 true；否则返回 false。要求：程序中请使用函数 `Int.compare`（系统提供），不要使用 `<, =, >`。

```sml
datatype order = GREATER | EQUAL | LESS
case Int.compare(x1, x2) of
    GREATER => (*x1 > x2*)
    | EQUAL => (*x1 = x2*)
    | LESS => (*x1 < x2*)
```

-   code

```sml
fun binarySearch(Empty, x) = false
    | binarySearch(Node(L, v, R), x) = case Int.compare(x, v) of
            GREATER => binarySearch(R, x)
            | LESS => binarySearch(L, x)
            | EQUAL => true;
```

### 4

一棵 `minheap` 树定义为：

1. t is Empty；
2. t is a Node(L, x, R), where R, L are minheaps and value(L), value(R) >= x（value(T)函数用于获取树 T 的根节点的值）

编写函数 `treecompare`，`SwapDown` 和 `heapify`：

```sml
treecompare: tree * tree -> order
(*when given the two trees, returns a value of type order, based on which tree has a larger value at the root node.*)

SwapDown: tree -> tree
(*REQUIRES the subtrees of t are both minheaps
 *ENSURES swapDown(t) = if t is Empty or all of t's immediate children are empty then just return t, otherwise returns a minheap which contains exactly the elements in t.
*)

heapify: tree -> tree
(*given an arbitrary tree t, evaluates to a minheap with exactly the elements of t.*)
```

分析 `SwapDown` 和 `heapify` 两个函数的 `work` 和 `span`。

- code final

```sml
datatype tree = Empty | Node of tree * int * tree;

(*
 * treecompare: tree * tree -> order
 * REQUIRES: two normal trees
 * ENSURES: when given the two trees, returns a value of type order, based on which tree has a larger value at the root node.
 *)
fun treecompare(Empty, Empty): order = EQUAL
    | treecompare(t1, Empty) = LESS
    | treecompare(Empty, t2) = GREATER
    | treecompare(Node(l1, x, r1), Node(l2, y, r2)) = case Int.compare(x, y) of
            GREATER => GREATER
            | _ => LESS;

(*
 * SwapDown: tree -> tree
 * REQUIRES: a normal tree
 * ENSURES: SwapDown(t) = if t is Empty or all of t's immediate children are empty then just return t, otherwise returns a minheap which contains exactly the elements in t.
 *)
fun SwapDown(Empty) = Empty
    | SwapDown(Node(Empty, x, Empty)) = Node(Empty, x, Empty)
    | SwapDown(Node(t1, x, t2)) = 
        if treecompare(t1, t2) = LESS then
            let
                val Node(l1, v1, r1) = t1
            in
                if (x > v1) then Node(SwapDown(Node(l1, x, r1)), v1, t2)
                else Node(t1, x, t2)
            end
        else
            let
                val Node(l2, v2, r2) = t2
            in
                if (x > v2) then Node(t1, v2, SwapDown(Node(l2, x, r2)))
                else Node(t1, x, t2)
            end;

(*
 * trav: 'a tree -> 'a list
 * REQUIRES: a normal tree
 * ENSURES: 
 *)
fun trav(Empty): int list = []
    | trav(T:tree): int list = 
        let
            val Node(l, x, r) = T
        in
            trav(l) @ (x::trav(r))
        end;

(*
 * heapify: tree -> tree
 * REQUIRES: a normal tree
 * ENSURES: given an arbitrary tree t, evaluates to a minheap with exactly the elements of t.
 *)
fun heapify(Empty) = Empty
    | heapify(Node(l, x, r)) = SwapDown(Node(SwapDown(l), x, SwapDown(r)));

fun split(L:int list) = 
        let
            val left = List.take(L, length(L) div 2)
            val right = List.drop(L, length(L) div 2)
        in
            if (length(right) = 0) then (left, 0, right)
            else (left, hd(right), List.drop(right, 1))
        end;

fun listToTree([]:int list): tree = Empty
    | listToTree((x:int)::[]): tree = Node(Empty, x, Empty)
    | listToTree((x:int)::L): tree =
        let
            val (L1:int list, y, L2:int list) = split(x::L)
        in
            Node(listToTree(L1), y, listToTree(L2))
        end;

val test = [~1, ~2, 3, 4, 5, 6, 7];
val tmp1 = heapify(listToTree(test));
val tmp2 = trav(tmp1);
```
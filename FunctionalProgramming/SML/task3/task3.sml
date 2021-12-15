(*树类型定义*)
datatype tree = Empty | Node of tree * int * tree;

(*树的中序遍历*)
fun trav Empty = []
    | trav(Node(t1, x, t2)) = trav t1 @ (x::trav t2);

(*split: int list -> int list * int list*)
(*将 list 拆分为长度相差小于 1 的两个 list*)
(* fun split [] = ([], 0, [])
    | split [x] = ([], x, [])
    | split (x::L) =
        let
            val(A, y, B) = split L
        in
            if length(A) > length(B) then (A, x, y::B)
            else (y::A, x, B)
        end; *)
fun split (L:int list)=
        let
            val left = List.take(L, length(L) div 2)
            val right = List.drop(L, length(L) div 2)
        in 
            if (length(right) = 0) then (left, 0, right)
            else (left, hd(right), List.drop(right, 1))
        end;

fun listToTree [] = Empty
    | listToTree (x::[]) = Node(Empty, x, Empty)
    | listToTree (x::L) =
        let 
            val (L1, y, L2) = split(x::L)
        in 
            Node(listToTree(L1), y, listToTree(L2))
        end;

fun revT(T) = case T of Empty => Empty
        | Node(ltree, x, rtree) => Node(revT(rtree), x, revT(ltree));

fun binarySearch(Empty, x) = false
    | binarySearch(Node(L, v, R), x) = case Int.compare(x, v) of
            GREATER => binarySearch(R, x)
            | LESS => binarySearch(L, x)
            | EQUAL => true;

(*测试*)
val l1 = [1, 2, 3, 4, 5, 6, 7];
val t1 = listToTree(l1);
trav t1;

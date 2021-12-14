datatype tree = Empty | Node of tree * int * tree;
fun treecompare(Empty, Empty): order = EQUAL
    | treecompare(t1, Empty) = LESS
    | treecompare(Empty, t2) = GREATER
    | treecompare(Node(l1, x, r1), Node(l2, y, r2)) = case Int.compare(x, y) of
            GREATER => GREATER
        | _ => LESS;

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

fun trav(Empty): int list = []
    | trav(T:tree): int list = 
        let
            val Node(l, x, r) = T
        in
            trav(l) @ (x::trav(r))
        end;

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
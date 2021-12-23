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
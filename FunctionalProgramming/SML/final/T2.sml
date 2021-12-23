(* isOdd: int -> bool *)
(* REQUIRES: x: int *)
(* ENSURES: return true is x is an odd number, else return false.*)
fun isOdd(0) = true
    | isOdd(1) = false
    | isOdd(x) = isOdd(x - 2);

(* enen: int list -> int list *)
(* REQUIRES: int list *)
(* ENSURES: return all the odd number(s) in the initial list and return them as a list. *)
fun even [] = []
    | even (x::L) =
        if isOdd(x) then x::even(L)
        else even(L);

(* a few tests *)
val test1 = [2, 4, 6, 7, 8];
val test2 = [1, 3, 5, 7];
val test3 = [1, 3, 5, 8];

val res1 = even(test1);
val res2 = even(test2);
val res3 = even(test3);
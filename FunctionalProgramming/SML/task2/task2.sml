(*interleave: int list * int list -> list*)
(*REQUIRES: int list, int list*)
(*ENSURES: 交错排列两个 list 中的元素，最后多余的元素直接连在后面。*)
fun interleave([], B) = B
    | interleave(A, []) = A
    | interleave(x::A, y::B) = [x, y] @ interleave(A, B);

(*测试*)
val L1 = [1, 2, 3, 4, 5, 6];
val L2 = [0, 0, 0];
val res = interleave(L1, L2);
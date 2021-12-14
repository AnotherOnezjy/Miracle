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
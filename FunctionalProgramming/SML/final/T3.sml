(* toInt: int -> int list -> int *)
(* REQUIRE: L为 b (b > 1) 进制数的 int list 表示 *)
(* ENSURE: toInt b L 为其相应的整数值 *)
fun toInt(base: int) (digits: int list): int  =
    case digits of
        [] => 0
        | d::digits => d + (base * (toInt base digits))

(* toBase: int -> int -> int list *)
(* REQUIRE: n为 10 进制，且 n >= 0 ,b > 1 *)
(* ENSURE: toBase b n 将十进制数 n 转换为 b 进制数的 int list 形式 *)
fun toBase (base: int) 0 = []
    | toBase (base: int) (n: int) =
        let
            val (q, r) = (n div base, n mod base)
        in
            r::(toBase base q)
        end;

(* convert: int * int -> int list -> int list *)
(* REQUIRE: b1,b2>1，L为b1进制数的int list表示 *)
(* ENSURE: convert(b1,b2) L 将 b1 进制数的列表形式
 * 转换为 b2 进制数列表形式
 * 即满足 toInt b2(convert(b1,b2) L) = toInt b1 L
 *)
fun convert (b1: int, b2: int) (L: int list) =
        let
            val n = toInt b1 L
        in
            toBase b2 n
        end;
datatype tree = Empty | Node of tree * int * tree;

(*树的递归表述：Node(t1, x, t2) (t1, t2: tree, x: integer)*)
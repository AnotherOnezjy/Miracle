#include "bits/stdc++.h"
using namespace std;

//Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode *next;

    ListNode() : val(0), next(nullptr) {}

    ListNode(int x) : val(x), next(nullptr) {}

    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

//Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode() : val(0), left(nullptr), right(nullptr) {}

    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}

    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

//Definition for a Node.
//class Node {
//public:
//    int val;
//    Node *left;
//    Node *right;
//    Node *next;
//
//    Node() : val(0), left(nullptr), right(nullptr), next(nullptr) {};
//
//    Node(int _val) : val(_val), left(nullptr), right(nullptr), next(nullptr) {}
//
//    Node(int _val, Node *_left, Node *_right, Node *_next) : val(_val), left(_left), right(_right), next(_next) {};
//};

//class Node {
//public:
//    int val;
//    vector<Node *> children;
//
//    Node() {}
//
//    Node(int _val) {
//        val = _val;
//    }
//
//    Node(int _val, vector<Node *> _children) {
//        val = _val;
//        children = _children;
//    }
//};

class Node {
public:
    int val;
    Node *prev;
    Node *next;
    Node *child;
};
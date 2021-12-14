// 字典树模板
// C++ Version
#include "bits/stdc++.h"
using namespace std;
struct trie {
    int nex[100000][26], cnt;
    bool exist[100000];//该结点结尾的字符串是否存在

    // insert: 插入字符串
    // s(char *): 待插入的字符串
    // l(int): 待插入字符串的长度
    void insert(char *s, int l) {
        int p = 0;
        for (int i = 0; i < l; i++) {
            int c = s[i] - 'a';
            if (!nex[p][c])//如果没有，就添加结点
                nex[p][c] = ++cnt;
            p = nex[p][c];
        }
        exist[p] = true;
    }

    // find: 查找字符串
    // s(char *): 待查找的字符串
    // l(int): 待查找字符串的长度
    bool find(char *s, int l) {
        int p = 0;
        for (int i = 0; i < l; i++) {
            int c = s[i] - 'a';
            if (!nex[p][c]) return false;
            p = nex[p][c];
        }
        return exist[p];
    }
};
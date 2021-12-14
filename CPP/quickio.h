#include <cstdio>
inline int read() {
    char ch;
    while ((ch = getchar()) < '0' || ch > '9');
    int res = ch ^ 48;
    while ((ch = getchar()) >= '0' && ch <= '9')
        res = res * 10 + (ch ^ 48);
    return res;
}
#include <cstdio>
#include <iostream>
#include <cstring>
#include <queue>
using namespace std;
const int N = 101010;

int cc = 0, to[N << 1], head[N], nex[N << 1];


void addedge(int x, int y){
    to[++cc] = y;
    nex[cc] = head[x];
    head[x] = cc;
}

void BFS(int now){

}

int main(){
    int n, m;
    scanf("%d%d", &n, &m);
    for(int i = 1; i <= m; i++){
        int x, y;
        scanf("%d%d", &x, &y);
        addedge(x, y);
    }
    BFS(1);
    int ans = dis[2];
    return 0;
}

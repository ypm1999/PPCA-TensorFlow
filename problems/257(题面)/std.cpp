#include <cstdio>
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;
const int N = 101010;

int cc = 0, to[N << 1], nex[N << 1], head[N], w[N << 1];
void addedge(int x, int y, int c){
	to[++cc] = y;
	nex[cc] = head[x];
	head[x] = cc;
	w[cc] = c;
}

long long val[N], fa[N], g1[N], g[N][2], f[N][2];
void dfs1(int x,int father) {
	g[x][0] = g[x][1] = val[x];
	g1[x] = x;
	fa[x] = father;
	for (int i = head[x]; i; i = nex[i])
		if (to[i] != father) {
			int y = to[i];
			dfs1(y, x);
			g[x][1] = max(g[x][1], g[x][1] + g[y][0] - 2 * w[i]);
			if (g[x][0] + g[y][1] - w[i] > g[x][1])
				g1[x] = y;
			g[x][1] = max(g[x][1], g[x][0] + g[y][1] - w[i]);
			g[x][0] += max(0ll, g[y][0] - 2 * w[i]);
		}
}

void dfs2(int x,int father,int z) {
	f[x][0] = f[x][1] = 0;
	if (g[x][0] <= 2 * z) {
		f[x][0] = max(f[x][0], f[father][0] + g[father][0] - 2 * z);
		f[x][1] = max(f[x][1], f[father][1] + g[father][0] - z);
	} else {
		f[x][0] = max(f[x][0], f[father][0] + g[father][0] - g[x][0]);
		f[x][1] = max(f[x][1], f[father][1] + g[father][0] - g[x][0] + z);
	}
	if (g1[father] == x) {
		int i;
		long long mx0 = val[father], mx1 = val[father];
		for (i = head[father]; i; i = nex[i])
			if (to[i] != fa[father] && to[i] != x) {
				int y = to[i];
				mx1 = max(mx1, mx1 + g[y][0] - 2 * w[i]);
				mx1 = max(mx1, mx0 + g[y][1] - w[i]);
				mx0 += max(0ll, g[y][0] - 2 * w[i]);
			}
		f[x][1] = max(f[x][1], mx1 + f[father][0] - z);
	} else {
		if (g[x][0] <= 2 * z)
			f[x][1] = max(f[x][1], f[father][0] + g[father][1] - z);
		else
			f[x][1] = max(f[x][1], f[father][0] + g[father][1] - g[x][0] + z);
	}
	for (int i = head[x]; i; i = nex[i])
		if (to[i] != father) dfs2(to[i], x, w[i]);
}



int main(){
	int n;
	cin >> n;
	for(int i = 1; i <= n; i++)
		scanf("%lld", &val[i]);
	for(int i = 1; i < n; i++){
		int x, y, c;
		scanf("%d%d%d", &x, &y, &c);
		addedge(x, y, c);
		addedge(y, x, c);
	}

	dfs1(1, 0);
	dfs2(1, 0, 0);
	for(int i = 1; i <= n; i++)
		printf("%lld\n", max(f[i][0] + g[i][1], f[i][1] + g[i][0]));
	return 0;
}

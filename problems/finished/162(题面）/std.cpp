#include <cstdio>
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;
const int N = 210;
const int M = 2010;
const int INF = 0x3f3f3f3f;

int mp[N][N], pos[M], f[2][N][N];

inline void update(int &x, const int &y){
	if(x > y)
		x = y;
}

int main(){
	int n, m;
	cin >> n >> m;
	memset(mp, 0x3f, sizeof(mp));
	for(int i = 1; i <= n; i++)
		for(int j = 1; j <= n; j++)
			scanf("%d", &mp[i][j]);
	for(int i = 1 ;i <= m; i++)
		scanf("%d", pos + i);

	// for(int k = 1; k <= n; k++)
	// 	for(int i = 1; i <= n; i++)
	// 		for(int j = 1; j <= n; j++)
	// 			update(mp[i][j], mp[i][k] + mp[k][j]);


	memset(f[0], 0x3f, sizeof(f[0]));
	f[0][1][2] = 0;
	pos[0] = 3;
	for(int i = 0; i < m; i++){
		int now = i & 1, nex = now ^ 1;
		memset(f[nex], 0x3f, sizeof(f[nex]));
		int res = INF;
		for(int x = 1; x <= n; x++)
			for(int y = x + 1; y <= n; y++)
				if(f[now][x][y] < INF){
					res = min(res, f[now][x][y]);
					int &pnow = pos[i], &pnex = pos[i + 1];
					// if(pnex != x && pnex != y)
						update(f[nex][x][y], f[now][x][y] + mp[pnow][pnex]);
					// if(pnex != x && pnex != pnow)
						update(f[nex][min(x, pnow)][max(x, pnow)], f[now][x][y] + mp[y][pnex]);
					// if(pnex != pnow && pnex != y)
						update(f[nex][min(y, pnow)][max(y, pnow)], f[now][x][y] + mp[x][pnex]);
				}
	}
	int ans = INF;
	for(int i = 1; i <= n; i++)
		for(int j = i + 1; j <= n; j++)
			ans = min(ans, f[m & 1][i][j]);
	cout << ans << endl;
    return 0;
}

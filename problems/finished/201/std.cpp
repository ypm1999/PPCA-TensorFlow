#include <cstdio>
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;
const int N = 1010101;
const int M = 510;
const int MOD = 1e9+7;

int fac[N], inv[N], h[N];
int n, m, cnt = 0, f[M][M], tmp[M][M];

int mypow(int x, int k){
	int res = 1;
	for(; k; k >>= 1, x = 1ll * x * x % MOD)
		if(k & 1)
			res = 1ll * res * x % MOD;
	return res;
}

void pre(int n){
	inv[0] = fac[0] = 1;
	for(int i = 1; i <= n; i++)
		fac[i] = 1ll * fac[i - 1] * i % MOD;
	inv[n] = mypow(fac[n], MOD - 2);
	for(int i = n - 1; i; i--)
		inv[i] = 1ll * inv[i + 1] * (i + 1) % MOD;
}

inline long long __C(int n, int m){
	return m > n ? 0ll : 1ll * fac[n] * inv[m] % MOD * inv[n - m] % MOD;
}

int dfs(int l, int r, int lasth){
	int now = ++cnt;

	int mn = 0x3f3f3f3f, id = -1;
	for(int i = l; i <= r; i++)
		if(h[i] < mn)
			mn = h[i], id = i;

	tmp[now][0] = f[now][0] = 1;
	int W = r - l + 1, H = mn - lasth;

	if(l == r){
		f[now][1] = H;
		return now;
	}
	int lc = 0, rc = 0;
	if(l < id)
		lc = dfs(l, id - 1, mn);
	if(r > id)
		rc = dfs(id + 1, r, mn);
	for(int i = 1; i <= m; i++)
		for(int j = 0; j <= i; j++)
			tmp[now][i] = (tmp[now][i] + 1ll * f[lc][j] * f[rc][i - j]) % MOD;

	for(int i = 1; i <= m; i++)
		for(int j = 0; j <= i; j++)
			f[now][i] = (f[now][i] + __C(H, j) * __C(W - (i - j), j) % MOD * fac[j] % MOD * tmp[now][i - j]) % MOD;
	return now;
}

int main(){
	pre(1000000);
	cin >> n >> m;
	for(int i = 1; i <= n; i++)
		cin >> h[i];
	memset(tmp, 0, sizeof(tmp));
	memset(f, 0, sizeof(f));
	f[0][0] = 1;
	int root = dfs(1, n, 0);
	cout << f[root][m] << endl;
    return 0;
}

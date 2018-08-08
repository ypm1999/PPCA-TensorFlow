#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 505, mod = 1e9 + 7;
int n, K, root;
int H[N], h[N], lc[N], rc[N], w[N];
ll dp[N][N], fac[1000005], f[N][N];
void up(ll &x, const ll &t) { x = (x + t) % mod; }
ll power(ll x, ll t)
{
	ll ret = 1;
	for(; t; t >>= 1, x = x * x % mod) if(t & 1) ret = ret * x % mod;
	return ret;
}
ll inv(ll x) { return power(x, mod - 2); }
ll C(int a, int b)
{
	if(a < b) return 0;
	return fac[a] * inv(fac[b]) % mod * inv(fac[a - b]) % mod;
}
ll calc(int a, int b, int K)
{
	if(a < K || b < K) return 0;
	ll ret = fac[K] * C(a, K) % mod * C(b, K) % mod;
	return ret;
}
void dfs(int u)
{
	f[u][0] = dp[u][0] = 1;
	if(!u) return;
	dfs(lc[u]);
	dfs(rc[u]);
	for(int i = 1; i <= K; ++i)
		for(int j = 0; j <= i; ++j)
			up(f[u][i], dp[lc[u]][j] * dp[rc[u]][i - j] % mod);
	for(int i = K; i >= 1; --i)
		for(int j = 0; j <= i; ++j)
			up(dp[u][i], f[u][j] * calc(H[u], w[u] - j, i - j) % mod);
}
int build(int l, int r)
{
	if(l > r) return 0;
	int p = l;
	for(int i = l; i <= r; ++i) if(h[i] < h[p]) p = i;
	lc[p] = build(l, p - 1);
	rc[p] = build(p + 1, r);
	H[lc[p]] = h[lc[p]] - h[p];
	H[rc[p]] = h[rc[p]] - h[p];
	w[p] = r - l + 1;
	return p;
}
int main(void)
{
	scanf("%d%d", &n, &K);
	fac[0] = 1;
	for(int i = 1; i <= n; ++i) scanf("%d", &h[i]), H[i] = h[i];
	for(int i = 1; i <= 1000000; ++i) fac[i] = fac[i - 1] * (ll)i % mod;
	root = build(1, n);
	dfs(root);
	printf("%lld\n", dp[root][K]);
	return 0;
}

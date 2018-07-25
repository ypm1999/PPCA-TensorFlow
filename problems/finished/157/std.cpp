#include <cstdio>
#include <iostream>
#include <cstring>
#include <queue>
using namespace std;
const int N = 101010;

int a[N], mx[N], mn[N];

int main(){
	int n = 0, sum = 0;
	cin >> n;
	for(int i = 1; i <= n; i++){
		scanf("%d", &a[i]);
		sum += a[i];
	}
	int ans = -0x3f3f3f3f;
	for(int i = 1; i <= n; i++){
		mx[i] = max(a[i], a[i] + mx[i - 1]);
		mn[i] = min(a[i], a[i] + mn[i - 1]);
		ans = max(ans, max(mx[i], sum - mn[i]));
	}
	cout << ans << endl;
    return 0;
}

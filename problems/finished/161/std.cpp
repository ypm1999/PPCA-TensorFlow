#include <cstdio>
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;
const int N = 101010;
typedef pair<int, int> pii;

pii lines[N * 3];

int main(){
	int n = 0, m = 0;
	cin >> n;
	long long sum = 0;
	for(int a, b, i = 1; i <= n; i++){
		scanf("%d%d", &a, &b);
		sum += abs(a - b);
		if(1ll * a * b >= 0 && (abs(a - b) <= abs(a) || abs(a) > abs(b)))
			continue;
		int h = abs(a - b) - abs(a);
		lines[m++] = make_pair(b - h, -1);
		lines[m++] = make_pair(b, 2);
		lines[m++] = make_pair(b + h, -1);
	}
	sort(lines, lines + m);
	long long y = 0, ans = 0, k = lines[0].second, x = lines[0].first;
	for(int i = 1; i < m; i++){
		int now = lines[i].first;
		y += k * (now - x);
		ans = min(ans, y);
		x = now;
		k += lines[i].second;
	}
	cout << ans + sum << endl;
    return 0;
}

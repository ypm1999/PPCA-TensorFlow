#include <cstdio>
#include <iostream>
#include <cstring>
#include <queue>
using namespace std;
const int N = 4010;

int f[2][N][3][3], a[N];

int main(){
	int n, m;
	cin >> n >> m;
	for(int i = 1; i <= n; i++)
		scanf("%d", &a[i]);
	memset(f[1], -0x3f, sizeof(int) * (9 * (m + 5)));
	f[1][0][0][0] = 0;
	f[1][1][1][1] = 0;
	f[1][1][2][2] = a[1];
	for(int i = 2; i <= n; i++){
		int now = i & 1, las = now ^ 1;
		memset(f[now], -0x3f, sizeof(int) * (9 * (m + 5)));
		f[now][0][0][0] = 0;
		f[now][0][1][0] = 0;
		f[now][0][2][0] = 0;
		for(int j = 1; j <= min(i, m); j++){
			for(int k = 0; k < 3; k++){
				int *fnow = f[now][j][k];
				int *flas1 = f[las][j - 1][k], *flas0 = f[las][j][k];
				fnow[0] = max(flas0[0], max(flas0[1], flas0[2]));
				fnow[2] = max(flas1[1], flas1[2]);
				fnow[1] = max(flas1[0], fnow[2]);
				fnow[2] += a[i];
				//cout << fnow[0] << " " << fnow[1] << " " << fnow[2] << endl;
			}
			//puts("");
		}
	}
	int ans = 0, nn = n & 1;
	for(int i = 0; i < 3; i++)
		for(int j = 0; j < 3; j++)
			if(!(i == 2 && j == 0))
				ans = max(ans, f[nn][m][i][j]);
	cout << ans << endl;
    return 0;
}

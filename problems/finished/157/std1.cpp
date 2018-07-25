#include <cstdio>
#include <iostream>
#include <cstring>
using namespace std;
const int N = 10101;

char s1[N], s2[N];
int nex[N], f[N][N / 5];

void getnext(char s[], int n){
	nex[1] = 0;
	for(int i = 2; i <= n; i++){
		int j = nex[i - 1];
		while(j != 0 && s[j + 1] != s[i])
			j = nex[j];
		if(s[j + 1] == s[i])
			j++;
		nex[i] = j;
	}
}

int main(){
	scanf("%s", s1 + 1);
	scanf("%s", s2 + 1);
	int n = strlen(s1 + 1);
	int m = strlen(s2 + 1);
	getnext(s2, m);
	memset(f, 0x3f, sizeof(f));
	f[0][0] = 0;
	for(int i = 0; i < n; i++)
		for(int j = 0; j < m && j <= i; j++)
			if(f[i][j] <= n){
				int &now = f[i][j];
//				cout << "::"<< i <<" "<< j <<" "<< now << endl;
				f[i + 1][j] = min(f[i + 1][j], now + 1);
				int pos = j;
				while(pos && s1[i + 1] != s2[pos + 1])
					pos = nex[pos];
				if(s1[i + 1] == s2[pos + 1])
					pos++;
//				cout << i + 1 <<" "<< pos <<" "<< now << endl;
				f[i + 1][pos] = min(f[i + 1][pos], now);
			}
	// for(int i = 0; i <= n; i++)
	// 	for(int j = 0; j <= m; j++)
	// 		cout << f[i][j] << " \n"[j == m];
	int ans = 0x3f3f3f3f;
	for(int i = 1; i < m; i++)
		ans = min(ans, f[n][i]);
	cout << ans << endl;
    return 0;
}

#include <cstdio>
#include <iostream>
#include <cstring>
#include <queue>
using namespace std;
const int N = 101010;

char s1[N], s2[N];
int cnt = 1, son[N][26], fail[N], f[2][N / 2];
bool num[N];

int insert(char s[], int n){
	int now = 1;
	for(int i = 1; i <= n; i++){
		int c = s[i] - 'a';
		if(!son[now][c])
			son[now][c] = ++cnt, memset(son[cnt], 0, sizeof(son[cnt]));
		now = son[now][c];
	}
	num[now]++;
	return now;
}

void buildAc(){
	queue<int> Q;
	Q.push(1);
	fail[1] = 0;
	while(!Q.empty()){
		int now = Q.front();
		Q.pop();
		for(int i = 0; i < 26; i++)
			if(son[now][i]){
				int y = fail[now];
				if(!son[y][i])
					y = 1;
				fail[son[now][i]] = son[y][i];
				Q.push(son[now][i]);
			}
			else
				son[now][i] = son[fail[now]][i];
	}
}

int main(){
	scanf("%s", s1 + 1);
	scanf("%s", s2 + 1);
	int n = strlen(s1 + 1);
	int m = strlen(s2 + 1);
	cnt = 1;
	memset(son[1], 0, sizeof(son[1]));
	for(int i = 0; i < 26; i++)
		son[0][i] = 1;
	insert(s2, m);
	buildAc();
	f[0][1] = 0;
	for(int i = 0; i < n; i++){
		int c = s1[i + 1] - 'a';
		int now = i & 1, nex = now ^ 1;
		memset(f[nex], 0x3f, sizeof(*f[nex]) * (m + 2));
		for(int j = 1; j < cnt; j++)
			if(f[now][j] <= n){
				f[nex][j] = min(f[nex][j], f[now][j] + 1);
				f[nex][son[j][c]] = min(f[nex][son[j][c]], f[now][j]);
			}
	}
	// for(int i = 1; i <= n; i++)
	// 	for(int j = 1; j <= cnt; j++)
	// 		cerr << f[i][j] << " \n"[j == cnt];
	int ans = 0x3f3f3f3f;
	for(int i = 1; i < cnt; i++)
		ans = min(ans, f[n & 1][i]);
	cout << ans << endl;
    return 0;
}

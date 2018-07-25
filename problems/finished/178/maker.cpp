#include <cstdio>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cassert>
using namespace std;

inline int random(int l, int r){
	assert(l <= r);
	return rand() % (r - l + 1) + l;
}

inline int random(int r){
	return random(1, r);
}

const int N = 101010;
char a[N], b[N];

void make(int n, int m){
	int k = random(10);
	for(int i = 1; i <= n; i++)
		a[i] = rand() % k + 'a';
	for(int i = 1; i <= m; i++)
		b[i] = rand() % k + 'a';
	for(int i = (n + m - 1) / m; i ; i--){
		int pos = random(1, n - m + 1);
		memcpy(a + pos, b + 1, m);
	}
	a[n + 1] = b[m + 1] = 0;
	puts(a + 1);
	puts(b + 1);
}

double F(int i, int n){
	double x = 1.0 * i / n;
	return x * x;
}

int main(){
	srand(time(0));
	string name = "./data/Necklace";
	string std = "./std";
	int num = 10;
	int maxn = 10000;
	int maxm = 1000;
	system("rm -rf ./data && mkdir data");
	for(int i = 1; i <=  num; i++){
		string fileIn = name + to_string(i) + ".in";
		string fileOut = name + to_string(i) + ".out";
		int basen = maxn * F(i, num);
		int basem = maxm * F(i, num);
		if(i % 2)
			basem = basem % 13 + 10;
		int n = random(basen * 0.9, basen);
		int m = random(basem * 0.9, basem);
		cerr << n << " " << m << endl;
		freopen(fileIn.c_str(), "w", stdout);
		make(n, m);
		fclose(stdout);
		system((std + " < " + fileIn + " > " + fileOut).c_str());
		cerr << "Make data: " + name + to_string(i) << " finished" << endl;
	}

    return 0;
}

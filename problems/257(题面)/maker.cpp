#include <cstdio>
#include <iostream>
#include <fstream>
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

void make(int n, int m){
	cout << n << " " << m << endl;
	int limt = 1e3;
	for(int i = 1; i <= n; i++)
		for(int j = 1; j <= n; j++)
			printf("%d%c", random(0,limt), " \n"[j == n]);
	for(int i = 1; i <= m; i++)
		printf("%d%c", random(n), " \n"[i == m]);
}

double F(int i, int n){
	double x = 1.0 * i / n;
	return x * x;
}

#define DEBUG
int main(){
	srand(time(0));
	string name = "./data/door";
	string std = "./std";
	int num = 10;
	int _maxn = 200;
	int _maxm = 1000;
	system("rm -rf ./data && mkdir data");
#ifdef DEBUG
	system("make std && make std1");
#endif
	for(int i = 1; i <=  num; i++){
		string fileIn = name + to_string(i) + ".in";
		string fileOut = name + to_string(i) + ".out";
		int maxn = F(i, num) * _maxn;
		int maxm = F(i, num) * _maxm;
		int n = random(maxn * 0.95, maxn);
		int m = random(maxm * 0.95, maxm);
		//cerr << n << endl;
		freopen(fileIn.c_str(), "w", stdout);
		make(n, m);
		fclose(stdout);
		system((std + " < " + fileIn + " > " + fileOut).c_str());
#ifdef DEBUG
		system(("./std < " + fileIn + " > std.out").c_str());
		system(("./std1 < " + fileIn + " > std1.out").c_str());
		cerr << "runing " << i << "   ";
		ifstream ans1("std.out"), ans2("std1.out");
		int a1, a2;
		ans1 >> a1;
		ans2 >> a2;
		if(a1 != a2){
			cerr << "error at " << i << endl;
			break;
		}
		else
			cerr << "passed" << endl;
#endif
		//cerr << "Make data: " + name + to_string(i) << " finished" << endl;
	}

    return 0;
}

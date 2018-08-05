#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <algorithm>
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
	for(int i = 1; i <= n; i++)
		printf("%d %d\n", random(1e6), random(m));
	// int last = 1e6;
	// for(int i = 1; i <= n; i++)
	// 	printf("%d %d\n",last -= rand() % 11 - 1, random(max(1, m / 1000)));
}

double F(int i, int n){
	double x = 1.0 * i / n;
	return x * x;
}

//#define DEBUG
int main(){
	srand(time(0));
	string name = "./data/shelf";
	string std = "./std";
	int l = 1;
	int r = 10;
	int num = r - l + 1;
	int _maxn = 1e5;
	int _maxm = 1e9;
	system("rm -rf ./data && mkdir data");
#ifdef DEBUG
	system("make std && make std1");
	system("rm -rf data.log");
#endif
	for(int i = l; i <= r; i++){
		string fileIn = name + to_string(i) + ".in";
		string fileOut = name + to_string(i) + ".out";
		int maxn = F(i - l + 1, num) * _maxn;
		int maxm = F(i - l + 1, num) * F(i - l + 1, num) * _maxm;
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
		system("diff std.out std1.out >> data.log");
#endif
		//cerr << "Make data: " + name + to_string(i) << " finished" << endl;
	}
    return 0;
}

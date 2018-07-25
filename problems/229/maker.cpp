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
	system("rm -rf data.log");
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
		system("diff std.out std1.out >> data.log");
#endif
		//cerr << "Make data: " + name + to_string(i) << " finished" << endl;
	}
    return 0;
}

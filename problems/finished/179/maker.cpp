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

void make(int n){
	cout << n << endl;
	for(int i = 1; i <= n; i++)
		printf("%d\n", random(2e9 + 1) - (int)1e9 - 1);
}

double F(int i, int n){
	double x = 1.0 * i / n;
	return x * x;
}

int main(){
	srand(time(0));
	string name = "./data/sequence";
	string std = "./std";
	int num = 10;
	int maxn = 500000;
	system("rm -rf ./data && mkdir data");
	for(int i = 1; i <=  num; i++){
		string fileIn = name + to_string(i) + ".in";
		string fileOut = name + to_string(i) + ".out";
		int n = random(maxn * 0.9, maxn);
		//cerr << n << endl;
		freopen(fileIn.c_str(), "w", stdout);
		make(n);
		fclose(stdout);
		system((std + " < " + fileIn + " > " + fileOut).c_str());
		system(("time ./std < " + fileIn + " > std.out").c_str());
		system(("time python ./std.py < " + fileIn + " > std1.out").c_str());
		cerr << "runing " << i << endl;
		ifstream ans1("std.out"), ans2("std1.out");
		int a1, a2;
		ans1 >> a1;
		ans2 >> a2;
		if(a1 != a2){
			cerr << "error at " << i << endl;
			break;
		}
		//cerr << "Make data: " + name + to_string(i) << " finished" << endl;
	}

    return 0;
}

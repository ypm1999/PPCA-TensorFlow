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
	int m = random(n * 0.8, n);
	cout << n << " " << m << endl;
	for(int i = 1; i <= n; i++)
		printf("%d%c", random(2e5), " \n"[i == n]);
}
int main(){
	srand(time(0));
	string name = "./data/sleep";
	string std = "./std";
	int num = 10;
	int _maxn = 4000;
	system("rm -rf ./data && mkdir data && rm -rf data.log");
	system("make std && make std1");
	for(int i = 1; i <=  num; i++){
		string fileIn = name + to_string(i) + ".in";
		string fileOut = name + to_string(i) + ".out";
		int maxn = _maxn * i / num;
		int n = random(maxn * 0.95, maxn);
		//cerr << n << endl;
		freopen(fileIn.c_str(), "w", stdout);
		make(n);
		fclose(stdout);
		system((std + " < " + fileIn + " > " + fileOut).c_str());
		system(("./std < " + fileIn + " > std.out").c_str());
		system(("./std1 < " + fileIn + " > std1.out").c_str());
		system("diff std.out std1.out >> data.log");
		//cerr << "Make data: " + name + to_string(i) << " finished" << endl;
	}

    return 0;
}

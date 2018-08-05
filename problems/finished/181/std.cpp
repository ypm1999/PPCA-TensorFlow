#include <cstdio>
#include <iostream>
#include <queue>
#include <set>
using namespace std;
const int N = 101010;

int h[N], w[N];
long long f[N];
deque<int> Q;
multiset<long long> s;

void push(int pos){
	while(!Q.empty() && h[Q.back()] < h[pos]){
		int pos = Q.back();
		Q.pop_back();
		if(!Q.empty()){
			// cerr <<"erase:"<< f[Q.back()] + h[pos] << endl;
			s.erase(s.find(f[Q.back()] + h[pos]));
		}
	}
	if(!Q.empty()){
		// cerr << "insert:" << f[Q.back()] + h[pos] << endl;
		s.insert(f[Q.back()] + h[pos]);
	}
	Q.push_back(pos);
}

void pop(){
	int pos = Q.front();
	Q.pop_front();
	// cerr <<"erase:"<< f[pos] + h[Q.front()] << endl;
	s.erase(s.find(f[pos] + h[Q.front()]));
}

int main(){
	int n, m;
	cin >> n >> m;
	for(int i = 1; i <= n; i++)
		scanf("%d %d", &h[i], &w[i]);
	for(int i = 1; i <= n; i++)
		w[i] += w[i - 1];
	int pos = 1;
	f[1] = h[1];
	Q.push_back(1);
	for(int i = 2; i <= n; i++){
		// cerr << i << endl;
		push(i);
		// cerr << "push" << endl;
		while(w[i] - w[pos - 1] > m)
			pos++;
		while(Q.front() < pos)
			pop();
		// cerr <<"pop" << endl;
		f[i] = f[pos - 1] + h[Q.front()];
		if(!s.empty())
			f[i] = min(f[i], *s.begin());
	}
	// for(int i = 1; i <= n; i++)
	cout << f[n] << endl;
    return 0;
}

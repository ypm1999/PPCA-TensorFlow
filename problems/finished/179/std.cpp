#include<bits/stdc++.h>
using namespace std;

priority_queue<int> Q;

int main(){
    int n, x;
    cin >> n >> x;
    Q.push(x);
    long long ans = 0;
    for(int i = 2; i <= n; i++){
        scanf("%d", &x);
        if(Q.top() > x){
            ans += Q.top() - x;
            Q.pop();
            Q.push(x);
        }
        Q.push(x);
    }
    cout << ans << endl;
}

#include <iostream>
#include <cstdio>
#include <algorithm>
#include <queue>

using namespace std;

/*
CF13C

设$f_i(x)$表示$a_i$转到x，且$a_i$之前所有的数字转到x所需的代价。

$f_i(x) = |a_i - x| + min_{x' \leq x} f_{i-1} (x')$

举例子：$f_3(x) = |a_3 - x| + min_{x' \leq x} (|a_2-x'|+min_{x'' \leq x'} |a_1-x''|)$

可以发现这个是个下凸的函数。

分段：

$Task1: opt_{i-1} \leq a_i $时 $opt_i == a_i$

$$f_i(x) = |a_i-x|+min_{x' \leq x} f_{i-1}(x')$$

$$f_i(x) = a_i-min_{x' \leq x} f_{i-1}(x')$$

$Task1.1: x\leq opt_{i-1}$

$$f_i(x)=a_i-x+f_{i-1}(x)$$

$Task1.2: x == opt_{i-1}$

$$f_i(x) = a_i-x+f_{i-1}(opt_{i-1})$$
$f_{i-x}(opt_{i-1})$显然为常数

$Task1.3 x\geq opt_{i-1}$

$$f_i(x) = -a_i+x+f_{i-1}(opt_{i-1})$$

$Task2: a_i \leq opt_{i-1}$

$Task 2.1: x\leq a_i$

$$f_i(x) = a_i -x +f_{i-1} (x) == Task1.1$$

$Task 2.2 a_i \leq x \leq opt_{i-1} $

$$f_i(x) = x-a_i+f_{i-1}(x) $$

$Task 2.3 opt_{i-1} \leq x$

$$f_i(x) = x-a_i +f_{i-1}(opt_{i-1}) == Task1.3$$
*/

int n,a[500050];

long long ans=0;

priority_queue <int> q;

inline char gc()
{
    static char now[1<<16],*S,*T;
    if (T==S){T=(S=now)+fread(now,1,1<<16,stdin);if (T==S) return EOF;}
    return *S++;
}
inline int read()
{
    int x=0,f=1;char ch=gc();
    while(ch<'0'||ch>'9') {if (ch=='-') f=-1;ch=gc();}
    while(ch<='9'&&ch>='0') x=x*10+ch-'0',ch=gc();
    return x*f;
}

int main()
{
	n=read();
	while (n--)
	{
		int a;
		a=read();
		q.push(a);
		if (q.top()>=a)
		{
			ans+=q.top()-a;
			q.pop();
			q.push(a);
		}
	}
	printf("%lld",ans);
	return 0;
}

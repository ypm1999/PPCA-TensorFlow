#include <cstdio>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

long long n;

struct line{

    long long k;
    long long d;

}p[300003];

bool cmp(line a,line b)
{
    return a.d<b.d ;
}

int main()
{
    long long base=0;
    long long m=0;
    scanf("%lld",&n);
    for(int i=0;i<n;i++)
    {
        long long a,b;
        scanf("%lld%lld",&a,&b);

        base+=abs(a-b);

        if(a*b>=0 && ( abs(a)>abs(b) || abs(b-a)<=abs(a) ) )
            continue;

        long long h=abs(a-b)-abs(a);
        p[m].d=b-h;
        p[m].k=-1;
        m++;
        p[m].d=b;
        p[m].k=2;
        m++;
        p[m].d=b+h;
        p[m].k=-1;
        m++;

    }

    sort(p,p+m,cmp);

    long long y=0, ans=0, slope=p[0].k, lx=p[0].d;

    for(long long i=1;i<m;i++)
    {
        long long x=p[i].d;
        y+=slope*(x-lx);

        if(y<ans)
            ans=y;

        lx=x;
        slope+=p[i].k;
    }
    printf("%lld",ans+base);

    return 0;
}

#include <cstdio>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <vector>

using namespace std;

int k,n;
int c[201][201];
int f[2][201][201];
int inf=1<<30;
int t[1001];

int main()
{

    scanf("%d%d",&k,&n);
    for(int i=1;i<=k;i++)
        for(int j=1;j<=k;j++)
            scanf("%d",&c[i][j]);

    for(int i=1;i<=n;i++)
        scanf("%d",&t[i]);

//    for(int x=1;x<=k;x++)
//        for(int y=1;y<=k;y++)
//            for(int z=1;z<=k;z++)
//            {
//                if(x==y||x==z||y==z)
//                    f[(n+1)%2][x][y][z]=inf;
//            }
    for(int x=1;x<=k;x++)
        for(int y=1;y<=k;y++)
            if(x==y||x==t[n]||y==t[n])
                f[(n+1)%2][x][y]=inf;

//    for(int i=n;i>0;i--)
//        for(int x=1;x<=k;x++)
//            for(int y=1;y<=k;y++)
//                for(int z=1;z<=k;z++)
//                {
//                    int ii=i%2;
//                    f[ii][x][y][z]=min (   min( f[1-ii][ t[i] ][y][z]+c[x][t[i]] , f[1-ii][x][ t[i] ][z] +c[y][t[i]] )  , f[1-ii][x][y][t[i]]+c[z][t[i]]  );
//                }

    t[0]=1;

    for(int i=n;i>0;i--)
        for(int x=1;x<=k;x++)
            for(int y=1;y<=k;y++)
            {
                int ii=i%2;
                if(x==y||x==t[i-1]||y==t[i-1])
                    f[ii][x][y]=inf;
                else
                    f[ii][x][y]=min(  min( f[1-ii][t[i-1]][y] + c[x][t[i]] ,
                                           f[1-ii][t[i-1]][x] + c[y][t[i]] )  ,
                                           f[1-ii][x][y]  +c[t[i-1]][t[i]]  );
            }

    printf("%d",f[1][2][3]);

    return 0;
}

//---NOTE---
//f(i,x,y,z):未来将修到i，三人在x，y，z时，完成之后任务需要的步数
//           ^ ^
//f(i,x,y,z)-->f(i+1,t,y,z)或f(i+1,x,t,z)或f(i+1,x,y,t)------的min值
//输出f(1,1,2,3)

//---优化---
//f(i,x,y):未来将修到i，其中两人在x,y另一人在t[i-1]上，完成之后任务需要的步数
//t[0]=1
//输出f(1,2,3)

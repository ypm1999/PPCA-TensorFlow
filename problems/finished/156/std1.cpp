#include<algorithm>
#include<iostream>
#include<cstring>
#include<cstdio>
#include<cmath>
using namespace std;
#define N 4010

int n,m,ans;
int a[N],f[2][N][3][3];

int Max(int a,int b,int c)
{
    if (a<b) a=b;
    if (a<c) a=c;
    return a;
}
int main()
{
    scanf("%d%d",&n,&m);
    for (int i=1;i<=n;++i) scanf("%d",&a[i]);
    memset(f,128,sizeof(f));
    for (int i=1;i<=n;++i) f[0][i][0][0]=0;
    for (int i=1;i<=m;++i)
    {
        memset(f[i&1],128,sizeof(f[i&1]));
        for (int j=i;j<=n;++j)
        {
            if (i==1&&j==1)
            {
                f[i&1][j][0][0]=0;
                f[i&1][j][1][1]=0;
                f[i&1][j][2][2]=a[1];
                continue;
            }
            for (int k=0;k<=2;++k)
            {
                f[i&1][j][0][k]=Max(f[i&1][j-1][0][k],f[i&1][j-1][1][k],f[i&1][j-1][2][k]);
                f[i&1][j][1][k]=max(f[(i-1)&1][j-1][0][k],f[(i-1)&1][j-1][1][k]);
                f[i&1][j][2][k]=max(f[(i-1)&1][j-1][1][k],f[(i-1)&1][j-1][2][k])+a[j];
            }
        }
    }
    for (int i=0;i<=2;++i)
        for (int j=0;j<=2;++j)
        {
            if (i==0&&j==2) continue;
            ans=max(ans,f[m&1][n][i][j]);
        }
    printf("%d\n",ans);
}

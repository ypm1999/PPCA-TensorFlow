#include<bits/stdc++.h>
typedef long long ll;
using namespace std;
int val[100010];
int ans[100010];
int val1[100010];//回到自身
int val2[100010];//没有回到自身
int val3[100010];//次优
int id[100010];//最后从哪个枝上面走了之后没有回到自身
vector<pair<int,int> >v[100010];
void dfs1(int s,int fa)
{
    val1[s]=val2[s]=val3[s]=val[s];
    for(int i=0;i<v[s].size();i++)
    {
        int t=v[s][i].first;
        int c=v[s][i].second;
        if(t==fa)
            continue;
        dfs1(t,s);
        int temp=max(val1[t]-2*c,0);
        val2[s]+=temp;
        val3[s]+=temp;
        if(val1[s]+val2[t]-c>val2[s])
        {
            val3[s]=val2[s];
            val2[s]=val1[s]+val2[t]-c;
            id[s]=t;
        }
        else if(val1[s]+val2[t]-c>val3[s])
            val3[s]=val1[s]+val2[t]-c;
        val1[s]+=temp;
    }
}
void dfs2(int s,int fa,int temp3,int temp4)
{//temp3表示向上走还要回来能得到的优势，temp4对应的是不回来的
    ans[s]=max(val1[s]+temp4,val2[s]+temp3);
    val2[s]+=temp3;
    val3[s]+=temp3;
    if(val2[s]<=val1[s]+temp4)//更新向上走了之后对应的结果
    {
        val2[s]=val1[s]+temp4;//这地方不更新val3[s]是因为一定用不到val3[s]了
        id[s]=fa;
    }
    else if(val3[s]<=val1[s]+temp4)
        val3[s]=val1[s]+temp4;
    val1[s]+=temp3;
    for(int i=0;i<v[s].size();i++)
    {
        int t=v[s][i].first;
        int c=v[s][i].second;
        if(t==fa)
            continue;
        int temp1=max(0,val1[s]-2*c-max(0,val1[t]-2*c));
        int temp2;
        if(id[s]==t)
            temp2=max(0,val3[s]-c-max(0,val1[t]-2*c));
        else temp2=max(0,val2[s]-c-max(0,val1[t]-2*c));
        dfs2(t,s,temp1,temp2);
    }
}
int main()
{
    int n;
    scanf("%d",&n);
    for(int i=1;i<=n;i++)
    {
        scanf("%d",&val[i]);
        v[i].clear();
    }
    int a,b,c;
    for(int i=1;i<n;i++)
    {
        scanf("%d%d%d",&a,&b,&c);
        v[a].push_back(make_pair(b,c));
        v[b].push_back(make_pair(a,c));
    }
    memset(id,-1,sizeof(id));
    dfs1(1,-1);
    dfs2(1,-1,0,0);
    for(int i=1;i<=n;i++)
        printf("%d ",ans[i]);
    return 0;
}

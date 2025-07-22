#include <stdio.h>
#define MaxInt 9999
#define MVNum 100
#define OK 1
#define ERROR -1
typedef int Status;
typedef char VerTexType;
typedef int ArcType;
typedef struct{ 
	VerTexType vexs[MVNum];	
	ArcType arcs[MVNum][MVNum];	
	int vexnum,arcnum;
}AMGraph;
typedef struct{
	int *base;	
	int front;
	int rear; 
}SqQueue;
Status InitQueue(SqQueue &Q){
	Q.base=new int[50];
	if(!Q.base) return ERROR;
	Q.front=Q.rear=0;
	return OK;
}
Status EnQueue(SqQueue &Q,int e){
	if((Q.rear+1)%50==Q.front) return ERROR;
	Q.base[Q.rear]=e;
	Q.rear=(Q.rear+1)%50;
	return OK;
}
Status DeQueue(SqQueue &Q,int &e){
	if(Q.front==Q.rear) return ERROR;
	e=Q.base[Q.front];
	Q.front=(Q.front+1)%50;
	return OK;
}
Status QueueEmpty(SqQueue Q){
	if(Q.front==Q.rear) return OK;
	else return ERROR;
}
int LocateVex(AMGraph G,int v){
	for(int i=0;i<G.vexnum;i++)
	{
		if(G.vexs[i]==v) return i;
	}
	return ERROR;
}
void CreateUND (AMGraph &G)
{
	ArcType w;
	printf("分别输入总顶点数，总边数："); 
	scanf(" %d,%d",&G.vexnum,&G.arcnum);
	getchar();
	printf("输入点的信息：\n");
	for(int i=0;i<G.vexnum;i++)
		scanf(" %c",&G.vexs[i]);

	for(int i;i<G.vexnum;i++)
		for(int j;j<G.vexnum;j++)
			G.arcs[i][j]=MaxInt;
	
	int m,n;//顶点下标 
	VerTexType v1,v2;
	printf("构造邻接矩阵:\n");
	for(int k=0;k<G.arcnum;k++)
	{
		scanf(" %c%c%d",&v1,&v2,&w);
		m=LocateVex(G,v1);n=LocateVex(G,v2);
		G.arcs[m][n]=w;
		G.arcs[n][m]=G.arcs[m][n];
	}
}
void print(AMGraph& G) {
	int i, j;
	printf("  ");
	for (i=0;i<G.vexnum;i++) {
		printf("%c ",G.vexs[i]);
	}
	printf("\n");
	for (i=0;i<G.vexnum;i++) {
		printf("%c ", G.vexs[i]);
		for (j=0;j<G.vexnum;j++) {
			if (G.arcs[i][j]==MaxInt)
				printf("∞ ");
			else
				printf("%d ", G.arcs[i][j]);
		}
		printf("\n");
	}
}
bool visited[MVNum]={false};
void BFS(AMGraph G,int v){
	SqQueue Q;
	printf("%c",G.vexs[v]);
	visited[v]=true;
	InitQueue(Q);
	EnQueue(Q,v);
	while(!QueueEmpty(Q)){
		int u;
		DeQueue(Q,v);
		for(int w=0;w<G.vexnum;w++)
			if(G.arcs[u][w] != MaxInt&&!visited[w]){
				printf("%c",G.vexs[w]);
				visited[w]=true;
				EnQueue(Q,w);
			}
	}
}
int main(){
	AMGraph G;CreateUND (G);print(G);BFS(G,1);return 0;
}

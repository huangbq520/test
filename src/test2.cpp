#include <stdio.h>
int main()
{
	int a[50];
	int i=0;
	for(i=0;scanf("%d",&a[i])!=EOF;i++)
	{
		scanf("%d",&a[i]);
	}
	int *p=a;
	for(p=a;p<a+i;p++){
		printf("%d",*p);
	} 
	
	return 0;
	
}

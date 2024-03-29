#include <stdio.h>
#include <math.h>

int main() {
	int m = 20, j;
	double x[6]={1.0, 2.0, 3.0, 4.0, 2.0, 1.0}, y[20];
	FILE *fp = fopen("y_n_output.txt", "w");
	y[0] = x[0];
	y[1] = -0.5*y[0]+x[1];
	fprintf(fp, "%lf\n%lf\n", y[0], y[1]);
	for (j = 2; j <= m - 1; j++) {
		if (j < 6) y[j] = -0.5*y[j-1]+x[j]+x[j-2];
		else if (j > 5 && j < 8) y[j] = -0.5*y[j-1]+x[j-2];
		else y[j] = -0.5*y[j-1];
		fprintf(fp, "%lf\n", y[j]);
	}
	fclose(fp);
	return 0;
}

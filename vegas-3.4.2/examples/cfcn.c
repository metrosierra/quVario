// file cfcn.c
#include <math.h>

double fcn(double x[], int dim)
{
    int i;
    double xsq = 0.0;
    for(i=0; i<dim; i++)
        xsq += x[i] * x[i] ;
    return exp(-100. * sqrt(xsq)) * pow(100.,dim);
}

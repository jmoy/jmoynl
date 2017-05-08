#ifndef __JMOY_CHEBYSHEV_H_
#define __JMOY_CHEBYSHEV_H_

void cheby_points(double *xs,int n);
void cheby_eval(double *coeffs,
                int n,
                double *xs,
                double *ys,
                int m);
void cheby_interp(double *ys,int n, double *coeffs);
#endif

#include <iostream>
#include <cstdio>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cassert>
using namespace std;

#include <fftw3.h>
#include <unistd.h>

#include "alloc.h"
#include "chebyshev.h"

double f(double x) {
  //return x*x;
  return sqrt(fabs(x));
}

double cheby_test(int n){
  /* Enumerate the Chebyshev points*/
  
  double *inter_x = alloc_vec(n);
  cheby_points(inter_x,n);

  /* Generate the Chebyshev interpolant*/
  double *inter_y = alloc_vec(n);;
#pragma omp parallel for
  for (int i=0;i<n;i++)
    inter_y[i] = f(inter_x[i]);
  double *coeffs = alloc_vec(n);
  cheby_interp(inter_y,n,coeffs);
  free_vec(inter_x);
  free_vec(inter_y);

  /* Evaluate and find the maximum error in function on 
     regular grid */

  int m = 210;

  double h = 2.0/m;
  double *xs = alloc_vec(m);
  double *ys = alloc_vec(m);
#pragma omp parallel for
  for (int i=0;i<m;i++){
    xs[i] = -1+i*h;
  }
  cheby_eval(coeffs,n,xs,ys,m);

  /* Compute error */
  double err = 0.0;
#pragma omp parallel for reduction(max:err)
  for (int i=0;i<m;i++){
    //printf("%.4g %.4g\n",t,y);
    double e = fabs(f(xs[i])-ys[i]);
    if (e>err)
      err = e;
  }
  free_vec(xs);
  free_vec(ys);
  return err;
}
  
int main(int argc,char *argv[]){
  if (argc<2){
    cerr <<"Usage: chebyshev [n1] [n2] ...\n";
    return 1;
  }
  int res = fftw_init_threads();
  assert(res==1);
  int ret_val = 0;

  if (access("wisdom.fftw",R_OK)==0)
    fftw_import_wisdom_from_filename("wisdom.fftw");

  while (--argc){
    int num_points = atoi(*++argv);
    if (num_points<1){
      cerr << "Num points must be positive. Skipping: "<<*argv<<'\n';
      ret_val = 1;
      continue;
    }
    double err = cheby_test(num_points);
    printf("%5d %.2e\n",num_points,err);
  }
  fftw_export_wisdom_to_filename("wisdom.fftw");
  return ret_val;
}

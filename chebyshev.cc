#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstring>
#include <cassert>

using namespace std;

#include <omp.h>
#include <fftw3.h>

#include "alloc.h"
#include "chebyshev.h"

//Chebyshev points of the first kind
void cheby_points(double *xs,int n)
{
  for (int i=0;i<n;i++)
    xs[i] = cos(M_PI * (i+0.5)/n);
}

// Evaluate the Chebyshev polynomial at a point
//   using Clenshaw's algorithm
void cheby_eval(double *coeffs,
                int n,
                double *xs,
                double *ys,
                int m)
{
#pragma omp parallel for
  for (int i=0;i<m;i++){
    double x = xs[i];
    double u0=0,u1=0,u2=0;
    for (int k=n;k>=0;k--){
      u2 = u1;
      u1 = u0;
      u0 = 2*x*u1-u2+coeffs[k];
    }
    ys[i] = 0.5*(coeffs[0]+u0-u2);
  }
}

void cheby_interp(double *ys,int n, double *coeffs)
{
  fftw_plan_with_nthreads(omp_get_max_threads());

  /* Try to get a plan with wisdom only*/
  fftw_plan plan
    = fftw_plan_r2r_1d(n,ys,coeffs,FFTW_REDFT10, FFTW_WISDOM_ONLY|FFTW_MEASURE);
  double *dummy = 0;
  if (plan==NULL){
    cerr << "No FFTW wisdom. Formulating plan.\n";
    dummy = alloc_vec(n);
    plan = fftw_plan_r2r_1d(n,dummy,coeffs,FFTW_REDFT10, FFTW_MEASURE);
    assert (plan!=NULL);
    memcpy(dummy,ys,sizeof(double)*n);
  }
  
  fftw_execute(plan);

  if (dummy!=0)
    free_vec(dummy);
  
  fftw_destroy_plan(plan);

  for (int k=0;k<n;k++){
    double scale = (k==0)?(2*n):n;
    coeffs[k] /= scale;
  }
}
    

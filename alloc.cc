#include <fftw3.h>

double *alloc_vec(size_t n){
  double *p = fftw_alloc_real(n);
  return p;
}

void free_vec(double *xs){
  fftw_free(xs);
}

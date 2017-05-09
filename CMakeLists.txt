cmake_minimum_required(VERSION 3.7)
project(Chebyshev)

add_executable(chtest chtest.cc alloc.cc chebyshev.cc)
target_link_libraries(chtest -fopenmp fftw3_omp fftw3 m)
target_compile_options(chtest PUBLIC -Wall -O3 -fopenmp -march=native -g)

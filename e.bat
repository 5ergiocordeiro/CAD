cls
g++ -c exercmat.c -pg -mms-bitfields -fopenmp -Ic:/arquiv~1/gnu/mingw/include -IOpenCL/inc -Ic:/software/openblas/include -Ic:/software/lapack/include
C:\Software\gfortran\bin\gfortran exercmat.o -pg -Lc:/software/openblas/lib -Lc:/software/lapack/lib -Lc:/software/gfortran/lib/gcc/mingw32/410~1.0 -lm -lgomp -lpthread -llapacke -lblas -Wl,-Bstatic -lopenblas -lgfortran
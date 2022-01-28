#include <stdio.h>
#include <iostream>
#include <math.h>
#define DIM 5000
#define ITER 1000
#define Tx 32 // could not execeed 1024
#define Ty 32

// function used to check cuda error
// reference: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void julia(float *a, float *b, float *c, float CR, float CI, float scale) {
    float R;
    float I;
    long int T = Tx * Ty;
    long int t = blockIdx.x * blockDim.x + threadIdx.x;
    long int n = DIM*DIM/T;
    long int r = (DIM*DIM)%T;
    long int l, h;
    // determine the range of iteration
    if (t < r) {
        l = t * (n+1);
        h = l + (n+1);
    } else {
        l = r * (n+1) + (t-r) * n;
        h = l + n;
    }
    for (long int i = l; i < h; i++){
        for (long int j=0; j<ITER; j++){
            R = a[i] * a[i] - b[i] * b[i] + CR;
            I = 2*a[i]*b[i] + CI;
            R = R/scale;
            I = I/scale;
            a[i] = R;
            b[i] = I;
            if ((R*R+I*I)/(scale*scale)>100.0) {
                c[i] = 255.0 * j/ITER;
                break;
            }
        }
    }
}

int main () {
    float x_min = -1.7e-0;
    float x_max =  1.7e-0;
    float y_min = -1.7e-0;
    float y_max =  1.7e-0;
    float scale =   100.0;
    float theta = M_PI * 0.919;
    float CR = cos(theta) * scale * scale;
    float CI = sin(theta) * scale * scale;
    float  *a = (float *)malloc(DIM*DIM*sizeof(float)); // real
    float  *b = (float *)malloc(DIM*DIM*sizeof(float)); // imaginary
    float  *c = (float *)malloc(DIM*DIM*sizeof(float)); // density
    float *da; // real
    float *db; // imaginary
    float *dc; // density
    long int m, n, p;
    FILE *ptr;

    // initialization
    printf("Initialization start...\n");
    for (long int i=0; i<DIM*DIM; i++){
        m = i%DIM;
        n = i/DIM;
        a[i] = x_min*scale + ((x_max-x_min)*scale)/(DIM-1)*m;
        b[i] = y_min*scale + ((y_max-y_min)*scale)/(DIM-1)*n;
        c[i] = 255.0;
    }

    gpuErrchk( cudaMalloc((void **)&da, DIM*DIM*sizeof(float)) );
    gpuErrchk( cudaMalloc((void **)&db, DIM*DIM*sizeof(float)) );
    gpuErrchk( cudaMalloc((void **)&dc, DIM*DIM*sizeof(float)) );
    gpuErrchk( cudaMemcpy(da, a, DIM*DIM*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(db, b, DIM*DIM*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(dc, c, DIM*DIM*sizeof(float), cudaMemcpyHostToDevice) );
    printf("Initialization done.\n");

    // main program
    printf("Program would be executed with %d cuda threads.\n", Tx*Ty);
    printf("Executing program...\n");
    julia<<<Ty, Tx>>>(da, db, dc, CR, CI, scale);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    printf("Main program finished.\n");
    
    // copy memory from device to host
    printf("Copy memory from device to host...\n");
    gpuErrchk( cudaMemcpy(a, da, DIM*DIM*sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(b, db, DIM*DIM*sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(c, dc, DIM*DIM*sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(da) );
    gpuErrchk( cudaFree(db) );
    gpuErrchk( cudaFree(dc) );
    printf("Copy memory done.\n");

    // save binary file
    printf("Saving data as binary file fractal.dat...\n");
    ptr = fopen("fractal.dat", "wb");
    fwrite(c, sizeof(float), DIM*DIM, ptr);
    fclose(ptr);
    
    // saving coordinates
    printf("Saving informations as text file coord.dat...\n");
    ptr = fopen("coord.dat", "w");
    fprintf(ptr, "%d\n%2.3f\n%2.3f\n%2.3f\n%2.3f\n%2.3f\n", DIM, theta, x_min, x_max, y_min, y_max);
    fclose(ptr);
    printf("Saving done.\n");

    return 0; 
}


#define STB_IMAGE_WRITE_IMPLEMENTATION
#define DIM 2000
#define ITER 512
#define Tx 64 // could not execeed 1024
#define Ty 64
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "stb_image_write.h"

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


__global__ void julia(float x_min, float x_max, float y_min, float y_max, char *c, float CR, float CI, float scale) {
    float a, b;
    int   p, q;
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
        p = i%DIM;
        q = i/DIM;
        a = x_min*scale + ((x_max-x_min)*scale)/(DIM-1)*p;
        b = y_min*scale + ((y_max-y_min)*scale)/(DIM-1)*q;
        for (long int j=0; j<ITER; j++){
            R = a * a - b * b + CR;
            I = 2*a*b + CI;
            R = R/scale;
            I = I/scale;
            a = R;
            b = I;
            if ((R*R+I*I)/(scale*scale)>100.0) {
                c[i] = 255 * j/ITER;
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
    char  *c = (char *)malloc(DIM * DIM * sizeof(char)); // density
    char  *dc; // density

    // initialization
    printf("Initialization start...\n");
    for (long int i=0; i<DIM*DIM; i++){
        c[i] = 255;
    }

    gpuErrchk( cudaMalloc((void **)&dc, DIM*DIM*sizeof(char)) );
    gpuErrchk( cudaMemcpy(dc, c, DIM*DIM*sizeof(char), cudaMemcpyHostToDevice) );
    printf("Initialization done.\n");

    // main program
    printf("Program would be executed with %d cuda threads.\n", Tx*Ty);
    printf("Executing program...\n");
    julia<<<Ty, Tx>>>(x_min, x_max, y_min, y_max, dc, CR, CI, scale);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    printf("Main program finished.\n");
    
    // copy memory from device to host
    printf("Copy memory from device to host...\n");
    gpuErrchk( cudaMemcpy(c, dc, DIM*DIM*sizeof(char), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(dc) );
    printf("Copy memory done.\n");

    // write png
    stbi_write_png("julia.png", DIM, DIM, 1, c, 0);

    printf("Done.\n");


    return 0; 
}


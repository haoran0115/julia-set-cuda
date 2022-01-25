#include <stdio.h>
#include <iostream>
#include <math.h>
#define DIM 5000
#define ITER 1000
#define Tx 1000 // could not execeed 1024
#define Ty 1000


__global__ void julia(double *a, double *b, double *c, double CR, double CI, double scale) {
    double R;
    double I;
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
            R = a[i]*a[i] - b[i]*b[i] + CR;
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
    static const double x_min = -1.7e-0;
    static const double x_max =  1.7e-0;
    static const double y_min = -1.7e-0;
    static const double y_max =  1.7e-0;
    static const double scale =   100.0;
    double theta = M_PI * 0.919;
    double CR = cos(theta) * scale * scale;
    double CI = sin(theta) * scale * scale;
    double  a[DIM*DIM]; // real
    double  b[DIM*DIM]; // imaginary
    double  c[DIM*DIM]; // density
    double *da;         // real
    double *db;         // imaginary
    double *dc;         // density
    long int m, n, p;
    FILE *ptr;

    // initialization
    printf("Initialization start...\n");
    for (long int i=0; i<DIM*DIM; i++){
        m = i%DIM;
        n = i/DIM;
        a[i] = x_min*scale + (x_max-x_min)*scale/(DIM-1)*m;
        b[i] = y_min*scale + (y_max-y_min)*scale/(DIM-1)*n;
        c[i] = 255.0;
    }

    cudaMalloc((void **)&da, sizeof(a));
    cudaMalloc((void **)&db, sizeof(b));
    cudaMalloc((void **)&dc, sizeof(c));
    cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, sizeof(b), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c, sizeof(c), cudaMemcpyHostToDevice);
    printf("Initialization done.\n");

    // main program
    printf("Program would be executed with %d cuda threads.\n", Tx*Ty);
    printf("Executing program...\n");
    julia<<<Ty, Tx>>>(da, db, dc, CR, CI, scale);
    cudaDeviceSynchronize();
    printf("Main program finished.\n");
    
    // copy memory from device to host
    printf("Copy memory from device to host...\n");
    cudaMemcpy(a, da, sizeof(a), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, db, sizeof(b), cudaMemcpyDeviceToHost);
    cudaMemcpy(c, dc, sizeof(c), cudaMemcpyDeviceToHost);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    printf("Copy memory done.\n");

    // save binary file
    printf("Saving data as binary file fractal.dat...\n");
    ptr = fopen("fractal.dat", "wb");
    fwrite(c, sizeof(c[0]), DIM*DIM, ptr);
    fclose(ptr);
    // saving coordinates
    printf("Saving informations as text file coord.dat...\n");
    ptr = fopen("coord.dat", "w");
    fprintf(ptr, "%d\n%2.3f\n%2.3f\n%2.3f\n%2.3f\n%2.3f\n", DIM, theta, x_min, x_max, y_min, y_max);
    fclose(ptr);
    printf("Saving done.\n");

    return 0; 
}


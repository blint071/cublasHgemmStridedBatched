#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include "cublas_v2.h" 
#include <cuda_fp16.h>
#include <iostream>
#include <sys/time.h>


//nvcc -lcublas cublas.c -o cublas.out

void main2()
{
int i,j,k,index;

// Linear dimension of matrices
int dim = 100;
int dim2 = 1;
int batch_count = 10000;
 
// Allocate host storage for batch_count A,B,C square matrices
half* h_A = (half*)malloc(sizeof(half) * dim2 * dim * batch_count);
half* h_B = (half*)malloc(sizeof(half) * dim * dim * batch_count);
half* h_C = (half*)malloc(sizeof(half) * dim * dim * batch_count);
    for(k=0; k<batch_count; k++) {
        for(j=0; j<dim; j++) {
                for(i=0; i<dim; i++) {
                index = i*dim + j + k*dim*dim;
                  //h_A[index] = index*index + 0.0f;
                  h_B[index] = index + 1.0f;
                  h_C[index] = 0.0f;
        }
    }
}

    for(k=0; k<batch_count; k++) {
        for(j=0; j<dim2; j++) {
                for(i=0; i<dim; i++) {
                index = i*dim + j + k*dim*dim;
                  h_A[index] = index*index + 0.0f;
        }
    }
}
 



half *d_A, *d_B, *d_C;
 
cudaMalloc(&d_A, sizeof(half) * dim2 * dim * batch_count);
cudaMalloc(&d_B, sizeof(half) * dim * dim * batch_count);
cudaMalloc(&d_C, sizeof(half) * dim * dim * batch_count);
 
cudaMemcpy(h_A,d_A,sizeof(half) * dim2 * dim * batch_count,cudaMemcpyDeviceToHost);
cudaMemcpy(h_B,d_B,sizeof(half) * dim * dim * batch_count,cudaMemcpyDeviceToHost);
cudaMemcpy(h_C,d_C,sizeof(half) * dim * dim * batch_count,cudaMemcpyDeviceToHost);

cublasHandle_t handle;
cublasCreate(&handle);
printf("hi");  
// Do the actual multiplication 
 
struct timeval t1, t2;
half alpha = 1.0f;  half beta = 1.0f;
for (int za=0 ; za<50000; za++)
{  
    
cublasHgemmStridedBatched(handle,
                              CUBLAS_OP_N, 
                              CUBLAS_OP_N,
                              dim, dim2, dim,
                              &alpha,
                              (const half*)d_A, dim,
                              dim2*dim,
                              (const half*)d_B, dim,
                              dim*dim,
                              &beta,
                              d_C, dim, 
                              dim*dim, 
                              batch_count);
}
 

cudaMemcpy(h_C,d_C,sizeof(half) * dim * dim * batch_count,cudaMemcpyDeviceToHost);
// Destroy the handle
cublasDestroy(handle);


cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
free(h_A);
free(h_B);
free(h_C);
}

int main(){
  
  main2();
  printf("Success!\n");
  return 0;
}
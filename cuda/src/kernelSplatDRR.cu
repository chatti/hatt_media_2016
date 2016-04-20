#define TPB 512

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


/////////Constant data//////////////////
__constant__ float dcamera[5];
__constant__ float dpcParams[1];
__constant__ float dinvCamera[5];
/////////////////////////////////////////

__global__ void ProjectPointCloud(float* PC, int* PC2D, float* phi)
{

	
	int N = (int)dpcParams[0];
	int w = (int)dcamera[0];
	int h = (int)dcamera[1];
	int u = 0;
	int v = 0;
    int j = blockIdx.x*blockDim.x + threadIdx.x; 

	
	float x   = PC[j];
	float y   = PC[j+N];
	float z   = PC[j+2*N];

	
	float dx  = 0.0;
	float dy  = 0.0;
	float dz  = 0.0;
	float f   = 0.0;

	
	f = phi[3]; float cx = cos(f); float sx = sin(f);
	f = phi[4]; float cy = cos(f); float sy = sin(f);
	f = phi[5]; float cz = cos(f); float sz = sin(f);
	f = dcamera[4];

	
	//Rotate
	dx = x*(cy*cz+sx*sy*sz) + y*(-cx*sz) + z*(cy*sx*sz-cz*sy);
	dy = x*(cy*sz-cz*sx*sy) + y*(cx*cz)  + z*(-sy*sz-cy*cz*sx);	
	dz = x*(cx*sy)          + y*sx       + z*(cx*cy);

	
	//Translate
	x = dx + phi[0];
	y = dy + phi[1];
	z = dz + phi[2];

	
	//Magnify and scale
	f = f/(f-z);
	x = x*f*dinvCamera[2];  //dinvCamera is the inverse of camera param.  Do this to avoid division operation
	y = y*f*dinvCamera[3];

	
	//Move to center of image
	x = x+(float)w*0.5;
	y = y+(float)h*0.5;
	u = (int)(x+0.5);
	v = (int)(y+0.5);
	
	
	PC2D[j] = (v*w) + u;
	
	//if(threadIdx.x==0)
	//{
	//	printf("u v j i p- %d %d %d %d %p\n",u,v,w*v+u,PC2D[j],PC2D);
	//}

	
}

cudaError_t SplatGPU(float* dphi, float* PC, int* PC2D, float* dPC, int* dPC2D, float* DRR, int W, int H, int N)
{
	
	cudaError_t status;
	int B = N/TPB;
	int T = TPB;
	
	ProjectPointCloud<<<B,T>>>(dPC, dPC2D, dphi);
	status = cudaGetLastError();
	if (status != cudaSuccess){printf("Problem running projection kernel!\n");return status;}
	status = cudaMemcpy(PC2D, dPC2D, N*sizeof(int), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess){printf("Copying 2D coords back to the host failed!\n");return status;}
	
	
	int j=0;
	int mxidx = W*H;
	for(int k=0; k<N; k++)
	{
		j = PC2D[k];
		if(j>=0 & j<mxidx){
			DRR[j] = DRR[j] + PC[k+3*N];
		}
	}
			
	return status;
}

cudaError_t SetConstantSplatData(float* camera, float* pcParams)
{
	cudaError_t status;

	float* inv;
	inv = (float*)malloc(5*sizeof(float));
	inv[0] = 1/camera[0];
	inv[1] = 1/camera[1];
	inv[2] = 1/camera[2];
	inv[3] = 1/camera[3];
	inv[4] = 1/camera[4];

	status = cudaMemcpyToSymbol(dcamera, camera, sizeof(float)*5);
	if(status!=cudaSuccess){printf("Error copying camera parameters to GPU!\n");return status;}	
	status = cudaMemcpyToSymbol(dpcParams, pcParams, sizeof(float));
	if(status!=cudaSuccess){printf("Error copying point cloud parameters to GPU!\n");return status;}
	status = cudaMemcpyToSymbol(dinvCamera, inv, sizeof(float)*5);
	if(status!=cudaSuccess){printf("Error copying inverse parameters to GPU!\n");return status;}
		
	return status;
}

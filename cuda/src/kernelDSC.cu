#define TPB 512

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

/////////Constant data//////////////////
__constant__ float d_camera1[12];
__constant__ float d_camera2[12];
__constant__ float d_invcamera1[12];
__constant__ float d_invcamera2[12];
__constant__ int   d_pcParams[1];
__constant__ float dT1[16];
__constant__ float dT2[16];
/////////////////////////////////////////

//Texture memory
texture<float, 2, cudaReadModeElementType> texX1;
texture<float, 2, cudaReadModeElementType> texX2;
texture<float, 2, cudaReadModeElementType> texGx1;
texture<float, 2, cudaReadModeElementType> texGx2;
texture<float, 2, cudaReadModeElementType> texGy1;
texture<float, 2, cudaReadModeElementType> texGy2;

__global__ void DSC_Mono(float* PC, float* T, float* cost)
{

	__shared__ float shrPxV[TPB];
	
	int i = threadIdx.x;
    int j = blockIdx.x*blockDim.x + i; 

	float x   = PC[j];
    float y   = PC[j+d_pcParams[0]];
    float z   = PC[j+2*d_pcParams[0]];
    float v   = PC[j+3*d_pcParams[0]];
	
	float dx  = 0.0;
	float dy  = 0.0;
	float dz  = 0.0;
	float f   = 0.0;

    f = d_camera1[4];

    //Spatial transform
    dx = x*T[0] + y*T[1] + z*T[2]  + T[3];
    dy = x*T[4] + y*T[5] + z*T[6]  + T[7];
    dz = x*T[8] + y*T[9] + z*T[10] + T[11];
	
	//Magnify and scale
    f = f/(f-dz);
    x = dx*f*d_invcamera1[2];  //dinvCamera is the inverse of camera param.  Do this to avoid division operation
    y = dy*f*d_invcamera1[3];

	
	//Move to center of image
    x = x+d_camera1[0]*0.5;
    //This needs to be inverted because images
    //y = d_camera1[1] - (y+d_camera1[1]*0.5);
    y = y+d_camera1[1]*0.5;
	
    shrPxV[i] = v*tex2D(texX1,x,y);

	__syncthreads();
    //Reduce the value
	if (i<256) { shrPxV[i] += shrPxV[i+256]; } __syncthreads();
	if (i<128) { shrPxV[i] += shrPxV[i+128]; } __syncthreads();
	if (i<64)  { shrPxV[i] += shrPxV[i+64];  } __syncthreads();
	if (i<32)  { shrPxV[i] += shrPxV[i+32];  } __syncthreads();
	if (i<16)  { shrPxV[i] += shrPxV[i+16];  } __syncthreads();
	if (i<8)   { shrPxV[i] += shrPxV[i+8];   } __syncthreads();
	if (i<4)   { shrPxV[i] += shrPxV[i+4];   } __syncthreads();
	if (i<2)   { shrPxV[i] += shrPxV[i+2];   } __syncthreads();
	if (i<1)   { cost[blockIdx.x] = shrPxV[i]+shrPxV[i+1]; }	
}

__global__ void GDSC_Mono(float* PC, float* T, float* cost)
{

    __shared__ float shrPxV[TPB];

    int i = threadIdx.x;
    int j = blockIdx.x*blockDim.x + i;

    float x   = PC[j];
    float y   = PC[j+d_pcParams[0]];
    float z   = PC[j+2*d_pcParams[0]];
    float v   = PC[j+3*d_pcParams[0]];
    float a   = PC[j+4*d_pcParams[0]];
    float b   = PC[j+5*d_pcParams[0]];
    float c   = PC[j+6*d_pcParams[0]];

    float dx  = 0.0;
    float dy  = 0.0;
    float dz  = 0.0;
    float f   = 0.0;

    f = d_camera1[4];

    //Spatial transform
    dx = x*T[0] + y*T[1] + z*T[2]  + T[3];
    dy = x*T[4] + y*T[5] + z*T[6]  + T[7];
    dz = x*T[8] + y*T[9] + z*T[10] + T[11];

    //Magnify and scale
    f = f/(f-dz);
    x = dx*f*d_invcamera1[2];  //dinvCamera is the inverse of camera param.  Do this to avoid division operation
    y = dy*f*d_invcamera1[3];


    //Move to center of image
    x = x+d_camera1[0]*0.5;
    //This needs to be inverted because images
    //y = d_camera1[1] - (y+d_camera1[1]*0.5);
    y = y+d_camera1[1]*0.5;

    //Spatial transform
    dx = a*T[0] + b*T[1] + c*T[2];
    dy = a*T[4] + b*T[5] + c*T[6];

    dx = sqrt(dx*dx + dy*dy);
    shrPxV[i] = dx*v*tex2D(texGx1,x,y) + dx*v*tex2D(texGx1,x,y);

    //shrPxV[i] = dx*v*tex2D(texGx1,x,y) + -dy*v*tex2D(texGy1,x,y);

    __syncthreads();
    //Reduce the value
    if (i<256) { shrPxV[i] += shrPxV[i+256]; } __syncthreads();
    if (i<128) { shrPxV[i] += shrPxV[i+128]; } __syncthreads();
    if (i<64)  { shrPxV[i] += shrPxV[i+64];  } __syncthreads();
    if (i<32)  { shrPxV[i] += shrPxV[i+32];  } __syncthreads();
    if (i<16)  { shrPxV[i] += shrPxV[i+16];  } __syncthreads();
    if (i<8)   { shrPxV[i] += shrPxV[i+8];   } __syncthreads();
    if (i<4)   { shrPxV[i] += shrPxV[i+4];   } __syncthreads();
    if (i<2)   { shrPxV[i] += shrPxV[i+2];   } __syncthreads();
    if (i<1)   { cost[blockIdx.x] = shrPxV[i]+shrPxV[i+1]; }
}


__global__ void DSC_Bi(float* PC, float* T1, float* T2, float* cost)
{

    __shared__ float shrPxV[TPB];

    int i = threadIdx.x;
    int j = blockIdx.x*blockDim.x + i;

    float x     = PC[j];
    float y     = PC[j+d_pcParams[0]];
    float z     = PC[j+2*d_pcParams[0]];
    float vx    = PC[j+3*d_pcParams[0]];
    float u     = 0.0;
    float v     = 0.0;

    float dx    = 0.0;
    float dy    = 0.0;
    float dz    = 0.0;
    float f     = 0.0;

    f = d_camera1[4];

    //Spatial transform
    dx = x*T1[0] + y*T1[1] + z*T1[2]  + T1[3];
    dy = x*T1[4] + y*T1[5] + z*T1[6]  + T1[7];
    dz = x*T1[8] + y*T1[9] + z*T1[10] + T1[11];

    //Magnify and scale
    f = f/(f-dz);
    u = dx*f*d_invcamera1[2];  //dinvCamera is the inverse of camera param.  Do this to avoid division operation
    v = dy*f*d_invcamera1[3];


    //Move to center of image
    u = u+d_camera1[0]*0.5;
    //This needs to be inverted because images are annoyingly going from up to down
    //v = d_camera1[1] - (v+d_camera1[1]*0.5);
    v = v+d_camera1[1]*0.5;

    f = d_camera2[4];

    //Spatial transform
    dx = x*T2[0] + y*T2[1] + z*T2[2]  + T2[3];
    dy = x*T2[4] + y*T2[5] + z*T2[6]  + T2[7];
    dz = x*T2[8] + y*T2[9] + z*T2[10] + T2[11];

    //Magnify and scale
    f = f/(f-dz);
    x = dx*f*d_invcamera2[2];  //dinvCamera is the inverse of camera param.  Do this to avoid division operation
    y = dy*f*d_invcamera2[3];


    //Move to center of image
    x = x+d_camera2[0]*0.5;
    //This needs to be inverted because images are annoyingly going from up to down
    //y = d_camera1[1] - (y+d_camera1[1]*0.5);
    y = y+d_camera2[1]*0.5;

    shrPxV[i] = vx*(tex2D(texX1,u,v)+tex2D(texX2,x,y));
    //shrPxV[i] = vx*(tex2D(texX1,u,v));

    __syncthreads();
    //Reduce the value
    if (i<256) { shrPxV[i] += shrPxV[i+256]; } __syncthreads();
    if (i<128) { shrPxV[i] += shrPxV[i+128]; } __syncthreads();
    if (i<64)  { shrPxV[i] += shrPxV[i+64];  } __syncthreads();
    if (i<32)  { shrPxV[i] += shrPxV[i+32];  } __syncthreads();
    if (i<16)  { shrPxV[i] += shrPxV[i+16];  } __syncthreads();
    if (i<8)   { shrPxV[i] += shrPxV[i+8];   } __syncthreads();
    if (i<4)   { shrPxV[i] += shrPxV[i+4];   } __syncthreads();
    if (i<2)   { shrPxV[i] += shrPxV[i+2];   } __syncthreads();
    if (i<1)   { cost[blockIdx.x] = shrPxV[i]+shrPxV[i+1]; }

    /*if(threadIdx.x==0 and blockIdx.x==0)
    {
        printf("%f %f %f %f\n",T1[0], T1[1], T1[2], T1[3]);
        printf("%f %f %f %f\n",T1[4], T1[5], T1[6], T1[7]);
        printf("%f %f %f %f\n",T1[8], T1[9], T1[10], T1[11]);
        printf("%f %f %f %f\n",T1[12], T1[13], T1[14], T1[15]);
        printf("____________\n");
        printf("%f %f %f %f\n",T2[0], T2[1], T2[2], T2[3]);
        printf("%f %f %f %f\n",T2[4], T2[5], T2[6], T2[7]);
        printf("%f %f %f %f\n",T2[8], T2[9], T2[10], T2[11]);
        printf("%f %f %f %f\n",T2[12], T2[13], T2[14], T2[15]);
    }*/

}



cudaError_t EntryCostFunctionDSC_BiPlane(float* d_pc, float* d_M1, float* d_M2, float* d_cost, float* h_cost, float* finalCost, int B, int T)
{
    float N = (float)(B*T);
    cudaError_t status;

    DSC_Bi<<<B,T>>>(d_pc, d_M1,d_M2, d_cost);
	status = cudaGetLastError();
    if (status != cudaSuccess){printf("Problem dash kernel (biplane)!\n");return status;}
	
    status = cudaMemcpy(h_cost, d_cost, B*sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess){printf("Copying cost back to the host failed!\n");return status;}
	
    finalCost[0]=0.0;
	for(int k=0; k<B; k++)
	{
        finalCost[0]+=h_cost[k];
	}
    finalCost[0] = -finalCost[0]/N;

	return status;
}

cudaError_t EntryCostFunctionDSC_MonoPlane(float* d_pc, float* d_M, float* d_cost, float* h_cost, float* finalCost, int B, int T)
{
    float N = (float)(B*T);
    cudaError_t status;

    DSC_Mono<<<B,T>>>(d_pc, d_M, d_cost);
    status = cudaGetLastError();
    if (status != cudaSuccess){printf("Problem dash kernel!\n");return status;}

    status = cudaMemcpy(h_cost, d_cost, B*sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess){printf("Copying cost back to the host failed!\n");return status;}

    finalCost[0]=0.0;
    for(int k=0; k<B; k++)
    {
        finalCost[0]+=h_cost[k];
    }
    finalCost[0] = -finalCost[0]/N;

    return status;
}

cudaError_t EntryCostFunctionGDSC_MonoPlane(float* d_pc, float* d_M, float* d_cost, float* h_cost, float* finalCost, int B, int T)
{
    float N = (float)(B*T);
    cudaError_t status;

    GDSC_Mono<<<B,T>>>(d_pc, d_M, d_cost);
    status = cudaGetLastError();
    if (status != cudaSuccess){printf("Problem Gdash kernel!\n");return status;}

    status = cudaMemcpy(h_cost, d_cost, B*sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess){printf("Copying cost back to the host failed!\n");return status;}

    finalCost[0]=0.0;
    for(int k=0; k<B; k++)
    {
        finalCost[0]+=h_cost[k];
    }
    finalCost[0] = -finalCost[0]/N;

    return status;
}

cudaError_t EntrySetupTextureDSC_X1(cudaArray* dI, cudaChannelFormatDesc chDesc)
{
	cudaError_t status;

    texX1.addressMode[0] = cudaAddressModeClamp;
    texX1.addressMode[1] = cudaAddressModeClamp;
    texX1.filterMode     = cudaFilterModeLinear;
    texX1.normalized     = false;

	status = cudaBindTextureToArray(texX1, dI,  chDesc);
	if (status != cudaSuccess){printf("Binding the texture failed!\n");return status;}
	return status;
}

cudaError_t EntrySetupTextureDSC_Gx1(cudaArray* dI, cudaChannelFormatDesc chDesc)
{
	cudaError_t status;

    texGx1.addressMode[0] = cudaAddressModeClamp;
    texGx1.addressMode[1] = cudaAddressModeClamp;
    texGx1.filterMode     = cudaFilterModeLinear;
    texGx1.normalized     = false;

	status = cudaBindTextureToArray(texGx1, dI,  chDesc);
	if (status != cudaSuccess){printf("Binding the texture failed!\n");return status;}
	return status;
}

cudaError_t EntrySetupTextureDSC_Gy1(cudaArray* dI, cudaChannelFormatDesc chDesc)
{
	cudaError_t status;

    texGy1.addressMode[0] = cudaAddressModeClamp;
    texGy1.addressMode[1] = cudaAddressModeClamp;
    texGy1.filterMode     = cudaFilterModeLinear;
    texGy1.normalized     = false;

	status = cudaBindTextureToArray(texGy1, dI,  chDesc);
	if (status != cudaSuccess){printf("Binding the texture failed!\n");return status;}
	return status;
}

cudaError_t EntrySetupTextureDSC_X2(cudaArray* dI, cudaChannelFormatDesc chDesc)
{
	cudaError_t status;

    texX2.addressMode[0] = cudaAddressModeClamp;
    texX2.addressMode[1] = cudaAddressModeClamp;
    texX2.filterMode     = cudaFilterModeLinear;
    texX2.normalized     = false;

    status = cudaBindTextureToArray(texX2, dI,  chDesc);
	if (status != cudaSuccess){printf("Binding the texture failed!\n");return status;}
	return status;
}

cudaError_t EntrySetupTextureDSC_Gx2(cudaArray* dI, cudaChannelFormatDesc chDesc)
{
	cudaError_t status;

    texGx2.addressMode[0] = cudaAddressModeClamp;
    texGx2.addressMode[1] = cudaAddressModeClamp;
    texGx2.filterMode     = cudaFilterModeLinear;
    texGx2.normalized     = false;

	status = cudaBindTextureToArray(texGx2, dI,  chDesc);
	if (status != cudaSuccess){printf("Binding the texture failed!\n");return status;}
	return status;
}

cudaError_t EntrySetupTextureDSC_Gy2(cudaArray* dI, cudaChannelFormatDesc chDesc)
{
	cudaError_t status;

    texGy2.addressMode[0] = cudaAddressModeClamp;
    texGy2.addressMode[1] = cudaAddressModeClamp;
    texGy2.filterMode     = cudaFilterModeLinear;
    texGy2.normalized     = false;

	status = cudaBindTextureToArray(texGy2, dI,  chDesc);
	if (status != cudaSuccess){printf("Binding the texture failed!\n");return status;}
	return status;
}

cudaError_t EntrySetConstantDataDSC(float* camera1, float* camera2, float pcsize)
{
	cudaError_t status;

	float inv1[12];
	float inv2[12];
    for( int k=0; k< 12; k++)
	{
        inv1[k] = 1/camera1[k];
        inv2[k] = 1/camera2[k];
	}	

	int pc[1];
    pc[0] = (int)pcsize;

	status = cudaMemcpyToSymbol(d_camera1, camera1, sizeof(float)*12);
	if(status!=cudaSuccess){printf("Error copying camera parameters to GPU!\n");return status;}	
	status = cudaMemcpyToSymbol(d_camera2, camera2, sizeof(float)*12);
	if(status!=cudaSuccess){printf("Error copying camera parameters to GPU!\n");return status;}	
	status = cudaMemcpyToSymbol(d_invcamera1, inv1, sizeof(float)*12);
	if(status!=cudaSuccess){printf("Error copying camera parameters to GPU!\n");return status;}	
	status = cudaMemcpyToSymbol(d_invcamera2, inv2, sizeof(float)*12);
	if(status!=cudaSuccess){printf("Error copying camera parameters to GPU!\n");return status;}

	status = cudaMemcpyToSymbol(d_pcParams, pc, sizeof(int));
	if(status!=cudaSuccess){printf("Error copying point cloud parameters to GPU!\n");return status;}

	return status;
}

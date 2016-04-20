#define TPB 512
#define TOTALRAYSTEPS 64
#define WIDTH 512
#define HEIGHT 512

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

cudaError_t CudaEntryFilterImage1(float* d_img, float* d_Gx, float* d_Gy);


texture<float, 3, cudaReadModeElementType> texCT;
/////////Constant data//////////////////
__constant__ float consCamera1[12];
__constant__ float consCamera2[12];
__constant__ float consInvCamera1[12];
__constant__ float consInvCamera2[12];
__constant__ float consRayParams1[4];
__constant__ float consRayParams2[4];
__constant__ float consInvCTparams[9];
__constant__ int   consBoundingBox1[4]; //{xmin, xmax, ymin, ymax}
__constant__ int   consBoundingBox2[4]; //{xmin, xmax, ymin, ymax}
__constant__ float consXrayMean1[1];
__constant__ float consXrayMean2[1];
__constant__ float consCam2CT1[16];
__constant__ float consCam2CT2[16];



__global__ void FilterImageGy1(float* img, float* Gy)
{
	__shared__ float shrCol[TPB];
    int w = (int)consCamera1[0];
    int h = (int)consCamera1[1];
	int i = threadIdx.x;	
	int j = w*i + blockIdx.x;

    shrCol[i] = img[j];

	__syncthreads();
    if(i>=1 & i < h-1)
	{
        shrCol[i] = 0.1065*shrCol[i-1]+0.7870*shrCol[i]+0.1065*shrCol[i+1];
        __syncthreads();
        Gy[j] = -0.5*shrCol[i-1]+0.5*shrCol[i+1];
	}
	else
	{
        Gy[j]=0.0;
	}


}

__global__ void FilterImageGx1(float* img, float* Gx)
{
	__shared__ float shrRow[TPB];	
    int w = (int)consCamera1[0];
	int i = threadIdx.x;	
	int j = w*blockIdx.x + i;


    shrRow[i] = img[j];
	__syncthreads();
    if(i>=1 & i < w-1)
	{
        shrRow[i] = 0.1065*shrRow[i-1]+0.7870*shrRow[i]+0.1065*shrRow[i+1];
        __syncthreads();
        Gx[j] = -0.5*shrRow[i-1]+0.5*shrRow[i+1];
	}
	else
	{
        Gx[j]=0.0;
	}	

}

__global__ void FilterImageGy2(float* img, float* Gy)
{
    __shared__ float shrCol[TPB];
    int w = (int)consCamera2[0];
    int h = (int)consCamera2[1];
    int i = threadIdx.x;
    int j = w*i + blockIdx.x;

    shrCol[i] = img[j];
    __syncthreads();
    if(i>=1 & i < h-1)
    {

        shrCol[i] = 0.1065*shrCol[i-1]+0.7870*shrCol[i]+0.1065*shrCol[i+1];
        __syncthreads();
        Gy[j] = -0.5*shrCol[i-1]+0.5*shrCol[i+1];
    }
    else
    {
        Gy[j]=0.0;
    }


}

__global__ void FilterImageGx2(float* img, float* Gx)
{
    __shared__ float shrRow[TPB];
    int w = (int)consCamera2[0];
    int i = threadIdx.x;
    int j = w*blockIdx.x + i;


    shrRow[i] = img[j];
    __syncthreads();
    if(i>=1 & i < w-1)
    {
        shrRow[i] = 0.1065*shrRow[i-1]+0.7870*shrRow[i]+0.1065*shrRow[i+1];
        __syncthreads();
        Gx[j] = -0.5*shrRow[i-1]+0.5*shrRow[i+1];
    }
    else
    {
        Gx[j]=0.0;
    }

}

__global__ void KernelReduceImage1(float* img, float* vec)
{
    __shared__ float shrMean[WIDTH];

    int i = threadIdx.x;
    int j = blockIdx.x;

    int gidx = ((int)consCamera1[0])*j + i;
    shrMean[i] = img[gidx];

    __syncthreads();
    //Reduce the value
    if (i<256) { shrMean[i] += shrMean[i+256]; } __syncthreads();
    if (i<128) { shrMean[i] += shrMean[i+128]; } __syncthreads();
    if (i<64)  { shrMean[i] += shrMean[i+64];  } __syncthreads();
    if (i<32)  { shrMean[i] += shrMean[i+32];  } __syncthreads();
    if (i<16)  { shrMean[i] += shrMean[i+16];  } __syncthreads();
    if (i<8)   { shrMean[i] += shrMean[i+8];   } __syncthreads();
    if (i<4)   { shrMean[i] += shrMean[i+4];   } __syncthreads();
    if (i<2)   { shrMean[i] += shrMean[i+2];   } __syncthreads();
    if (i<1)   { vec[j] = shrMean[i] + shrMean[i+1]; }
}

__global__ void KernelReduceImage2(float* img, float* vec)
{
    __shared__ float shrMean[WIDTH];

    int i = threadIdx.x;
    int j = blockIdx.x;

    int gidx = ((int)consCamera2[0])*j + i;
    shrMean[i] = img[gidx];

    __syncthreads();
    //Reduce the value
    if (i<256) { shrMean[i] += shrMean[i+256]; } __syncthreads();
    if (i<128) { shrMean[i] += shrMean[i+128]; } __syncthreads();
    if (i<64)  { shrMean[i] += shrMean[i+64];  } __syncthreads();
    if (i<32)  { shrMean[i] += shrMean[i+32];  } __syncthreads();
    if (i<16)  { shrMean[i] += shrMean[i+16];  } __syncthreads();
    if (i<8)   { shrMean[i] += shrMean[i+8];   } __syncthreads();
    if (i<4)   { shrMean[i] += shrMean[i+4];   } __syncthreads();
    if (i<2)   { shrMean[i] += shrMean[i+2];   } __syncthreads();
    if (i<1)   { vec[j] = shrMean[i] + shrMean[i+1]; }
}

__global__ void KernelNCC1(float* drr, float* xray, float* drrmean, float* nccvector, float* stdvector)
{
    __shared__ float shrNCC[TPB];
    __shared__ float shrStd[TPB];

    int i = threadIdx.x;
    int j = blockIdx.x;

    int gidx = ((int)consCamera1[0])*j + i;

    float mean = drrmean[0]*consInvCamera1[0]*consInvCamera1[1];


    float sd = drr[gidx]-mean;

    shrNCC[i] = sd*(xray[gidx] - consXrayMean1[0]);
    shrStd[i] = sd*sd;

    __syncthreads();
    //Reduce the value
    if (i<256) { shrNCC[i] += shrNCC[i+256]; shrStd[i] += shrStd[i+256]; } __syncthreads();
    if (i<128) { shrNCC[i] += shrNCC[i+128]; shrStd[i] += shrStd[i+128]; } __syncthreads();
    if (i<64)  { shrNCC[i] += shrNCC[i+64];  shrStd[i] += shrStd[i+64]; } __syncthreads();
    if (i<32)  { shrNCC[i] += shrNCC[i+32];  shrStd[i] += shrStd[i+32];  } __syncthreads();
    if (i<16)  { shrNCC[i] += shrNCC[i+16];  shrStd[i] += shrStd[i+16];  } __syncthreads();
    if (i<8)   { shrNCC[i] += shrNCC[i+8];   shrStd[i] += shrStd[i+8];   } __syncthreads();
    if (i<4)   { shrNCC[i] += shrNCC[i+4];   shrStd[i] += shrStd[i+4];   } __syncthreads();
    if (i<2)   { shrNCC[i] += shrNCC[i+2];   shrStd[i] += shrStd[i+2];   } __syncthreads();
    if (i<1)   { nccvector[j] = shrNCC[i]+shrNCC[i+1]; stdvector[j] = shrStd[i]+shrStd[i+1];}

}

__global__ void KernelNCC2(float* drr, float* xray, float* drrmean, float* nccvector, float* stdvector)
{
    __shared__ float shrNCC[TPB];
    __shared__ float shrStd[TPB];

    int i = threadIdx.x;
    int j = blockIdx.x;

    int gidx = ((int)consCamera2[0])*j + i;

    float mean = drrmean[0]*consInvCamera2[0]*consInvCamera2[1];


    float sd = drr[gidx]-mean;

    shrNCC[i] = sd*(xray[gidx] - consXrayMean2[0]);
    shrStd[i] = sd*sd;

    __syncthreads();
    //Reduce the value
    if (i<256) { shrNCC[i] += shrNCC[i+256]; shrStd[i] += shrStd[i+256]; } __syncthreads();
    if (i<128) { shrNCC[i] += shrNCC[i+128]; shrStd[i] += shrStd[i+128]; } __syncthreads();
    if (i<64)  { shrNCC[i] += shrNCC[i+64];  shrStd[i] += shrStd[i+64]; } __syncthreads();
    if (i<32)  { shrNCC[i] += shrNCC[i+32];  shrStd[i] += shrStd[i+32];  } __syncthreads();
    if (i<16)  { shrNCC[i] += shrNCC[i+16];  shrStd[i] += shrStd[i+16];  } __syncthreads();
    if (i<8)   { shrNCC[i] += shrNCC[i+8];   shrStd[i] += shrStd[i+8];   } __syncthreads();
    if (i<4)   { shrNCC[i] += shrNCC[i+4];   shrStd[i] += shrStd[i+4];   } __syncthreads();
    if (i<2)   { shrNCC[i] += shrNCC[i+2];   shrStd[i] += shrStd[i+2];   } __syncthreads();
    if (i<1)   { nccvector[j] = shrNCC[i]+shrNCC[i+1]; stdvector[j] = shrStd[i]+shrStd[i+1];}

}

__global__ void kernelRayCast1(float* DRR)
{

    int xstop = consBoundingBox1[0] + consBoundingBox1[1];
    int ystop = consBoundingBox1[2] + consBoundingBox1[3];

    int u   = blockDim.x*blockIdx.x + threadIdx.x;
    int v   = blockDim.y*blockIdx.y + threadIdx.y;
    int W = (int)consCamera1[0];
    int gidx = v*W+u;


    if( u >= consBoundingBox1[0] & u < xstop & v >= consBoundingBox1[2] & v < ystop )
    {

        //float SID    = consCamera1[4];       //SID
        float invSID = consInvCamera1[4];

        int steps  = (int)consRayParams1[2];

        //This are the locations of pixels, in physical coordinates
        //The origin is the center of the detector
        float x = consCamera1[2]*((float)u+0.5 - 0.5*consCamera1[0]);
        float y = consCamera1[3]*((float)v+0.5 - 0.5*consCamera1[1]);
        //This is the initial, bottom most, z, in physical coordinates
        float z = consRayParams1[0];

        //compute the change in x or y per change in z
        //float dz = l/s;                              //deltaZ = consRayParams1[1]
        float dz = consRayParams1[3];
        float dx = dz*x*invSID;
        float dy = dz*y*invSID;

        //These are the initial x and y locations in the physical space
        x = x - z*x*invSID;
        y = y - z*y*invSID;

        //Now each ray is defined. Each thread is a ray that travels through the volume
        float tx;
        float ty;
        float tz;

        float sum=0.0;

        for(int k=0; k<steps; k++)
        {
            x = x - dx;
            y = y - dy;
            z = z + dz;

            tx = x*consCam2CT1[0] + y*consCam2CT1[1] + z*consCam2CT1[2]  + consCam2CT1[3];
            ty = x*consCam2CT1[4] + y*consCam2CT1[5] + z*consCam2CT1[6]  + consCam2CT1[7];
            tz = x*consCam2CT1[8] + y*consCam2CT1[9] + z*consCam2CT1[10] + consCam2CT1[11];

            sum += tex3D(texCT,tx,ty,tz);

        }

        DRR[gidx] = sqrt(sum/(sqrt(dx*dx+dy*dy+dz*dz)*steps));
        //DRR[gidx]=1;
    }
    else
    {
        DRR[gidx] = 0;
    }

}

__global__ void kernelRayCast2(float* DRR)
{

    int xstop = consBoundingBox2[0] + consBoundingBox2[1];
    int ystop = consBoundingBox2[2] + consBoundingBox2[3];

    int u   = blockDim.x*blockIdx.x + threadIdx.x;
    int v   = blockDim.y*blockIdx.y + threadIdx.y;
    int W = (int)consCamera2[0];
    int gidx = v*W+u;


    if( u >= consBoundingBox2[0] & u < xstop & v >= consBoundingBox2[2] & v < ystop )
    {

        //float SID    = consCamera1[4];       //SID
        float invSID = consInvCamera2[4];
        int steps  = (int)consRayParams2[2];

        //This are the locations of pixels, in physical coordinates
        //The origin is the center of the detector
        float x = consCamera2[2]*((float)u+0.5 - 0.5*consCamera2[0]);
        float y = consCamera2[3]*((float)v+0.5 - 0.5*consCamera2[1]);
        //This is the initial, bottom most, z, in physical coordinates
        float z = consRayParams2[0];

        //compute the change in x or y per change in z
        //float dz = l/s;                              //deltaZ = consRayParams1[1]
        float dz = consRayParams2[3];
        float dx = dz*x*invSID;
        float dy = dz*y*invSID;

        //These are the initial x and y locations in the physical space
        x = x - z*x*invSID;
        y = y - z*y*invSID;

        //Now each ray is defined. Each thread is a ray that travels through the volume
        float tx;
        float ty;
        float tz;

        float sum=0.0;

        for(int k=0; k<steps; k++)
        {
            x = x - dx;
            y = y - dy;
            z = z + dz;

            tx = x*consCam2CT2[0] + y*consCam2CT2[1] + z*consCam2CT2[2]  + consCam2CT2[3];
            ty = x*consCam2CT2[4] + y*consCam2CT2[5] + z*consCam2CT2[6]  + consCam2CT2[7];
            tz = x*consCam2CT2[8] + y*consCam2CT2[9] + z*consCam2CT2[10] + consCam2CT2[11];

            sum += tex3D(texCT,tx,ty,tz);

        }

        DRR[gidx] = sum/(sqrt(dx*dx+dy*dy+dz*dz)*steps);
        //DRR[gidx]=1;
    }
    else
    {
        DRR[gidx] = 0;
    }

}

cudaError_t CudaEntryRayCast1(float* h_Cam2CT, int* h_BoundingBox, float* h_rayParams, float* d_DRR)
{
	cudaError_t status;

    status = cudaMemcpyToSymbol(consCam2CT1, h_Cam2CT, sizeof(float)*16);
    if(status != cudaSuccess){printf("Error copying transform to GPU");}
    status = cudaMemcpyToSymbol(consBoundingBox1, h_BoundingBox, sizeof(int)*4);
    if(status != cudaSuccess){printf("Error copying bounding box to GPU");}
    status = cudaMemcpyToSymbol(consRayParams1, h_rayParams, sizeof(float)*4);
    if(status != cudaSuccess){printf("Error copying ray params to GPU");}

	////////////////////////////////

    int blkdim=16;
    int boxWidth = WIDTH;
    int boxHeight = HEIGHT;
    dim3 B(boxWidth/blkdim,boxHeight/blkdim);
    dim3 T(blkdim,blkdim);
    kernelRayCast1<<<B,T>>>(d_DRR);
    status = cudaGetLastError();
    if(status != cudaSuccess){printf("Error running the ray casting kernel");};
	////////////////////////////////


	return status;
}

cudaError_t CudaEntryNCC1(float* h_Cam2CT, int* h_BoundingBox, float* h_rayParams, float* h_std, float h_xraystd,
                          float* d_DRR, float* d_xray, float* d_vector1, float* d_vector2, float* d_mean, float* d_ncc, float* d_std,
                          float* h_cost)
{
    cudaError_t status;

    status = cudaMemcpyToSymbol(consCam2CT1, h_Cam2CT, sizeof(float)*16);
    if(status != cudaSuccess){printf("Error copying transform to GPU");}
    status = cudaMemcpyToSymbol(consBoundingBox1, h_BoundingBox, sizeof(int)*4);
    if(status != cudaSuccess){printf("Error copying bounding box to GPU");}
    status = cudaMemcpyToSymbol(consRayParams1, h_rayParams, sizeof(float)*4);
    if(status != cudaSuccess){printf("Error copying ray params to GPU");}

    ////////////////////////////////
    int blkdim=16;
    int boxWidth = WIDTH;
    int boxHeight = HEIGHT;
    dim3 B1(boxWidth/blkdim,boxHeight/blkdim);
    dim3 T1(blkdim,blkdim);
    kernelRayCast1<<<B1,T1>>>(d_DRR);
    status = cudaGetLastError();
    if(status != cudaSuccess){printf("Error running the kernel");};
    ////////////////////////////////

    ////////////////////////////////

    //dim3 B2(boxHeight,boxWidth);
    //dim3 T2(blkdim,blkdim);
    KernelReduceImage1<<<HEIGHT,WIDTH>>>(d_DRR, d_vector1);
    KernelReduceImage1<<<1,HEIGHT>>>(d_vector1, d_mean);


    KernelNCC1<<<HEIGHT,WIDTH>>>(d_DRR, d_xray, d_mean, d_vector1, d_vector2);
    KernelReduceImage1<<<1,HEIGHT>>>(d_vector1, d_ncc);
    KernelReduceImage1<<<1,HEIGHT>>>(d_vector2, d_std);
    status = cudaGetLastError();
    if(status != cudaSuccess){printf("Error running the kernel");};
    ////////////////////////////////

    status = cudaMemcpy(h_cost, d_ncc, sizeof(float), cudaMemcpyDeviceToHost);
    if(status != cudaSuccess){printf("Error copying the cost function back the kernel");};

    status = cudaMemcpy(h_std, d_std, sizeof(float), cudaMemcpyDeviceToHost);
    if(status != cudaSuccess){printf("Error copying the std function back the kernel");};

    h_std[0] = sqrt(h_std[0]/(HEIGHT*WIDTH));
    h_cost[0] = -h_cost[0]/(HEIGHT*WIDTH*h_std[0]*h_xraystd);

    //status = cudaMemcpy(h_ncc, d_ncc, sizeof(float), cudaMemcpyDeviceToHost);
    //if(status != cudaSuccess){printf("Error copying the cost function back the kernel");};


    return status;
}

cudaError_t CudaEntryNCC2(float* h_Cam2CT, int* h_BoundingBox, float* h_rayParams, float* h_std, float h_xraystd,
                          float* d_DRR, float* d_xray, float* d_vector1, float* d_vector2, float* d_mean, float* d_ncc, float* d_std,
                          float* h_cost)
{
    cudaError_t status;

    status = cudaMemcpyToSymbol(consCam2CT2, h_Cam2CT, sizeof(float)*16);
    if(status != cudaSuccess){printf("Error copying transform to GPU");}
    status = cudaMemcpyToSymbol(consBoundingBox2, h_BoundingBox, sizeof(int)*4);
    if(status != cudaSuccess){printf("Error copying bounding box to GPU");}
    status = cudaMemcpyToSymbol(consRayParams2, h_rayParams, sizeof(float)*4);
    if(status != cudaSuccess){printf("Error copying ray params to GPU");}

    ////////////////////////////////
    int blkdim=16;
    int boxWidth = WIDTH;
    int boxHeight = HEIGHT;
    dim3 B1(boxWidth/blkdim,boxHeight/blkdim);
    dim3 T1(blkdim,blkdim);
    kernelRayCast2<<<B1,T1>>>(d_DRR);
    status = cudaGetLastError();
    if(status != cudaSuccess){printf("Error running the kernel");};
    ////////////////////////////////

    ////////////////////////////////

    //dim3 B2(boxHeight,boxWidth);
    //dim3 T2(blkdim,blkdim);
    KernelReduceImage2<<<HEIGHT,WIDTH>>>(d_DRR, d_vector1);
    KernelReduceImage2<<<1,HEIGHT>>>(d_vector1, d_mean);


    KernelNCC2<<<HEIGHT,WIDTH>>>(d_DRR, d_xray, d_mean, d_vector1, d_vector2);
    KernelReduceImage2<<<1,HEIGHT>>>(d_vector1, d_ncc);
    KernelReduceImage2<<<1,HEIGHT>>>(d_vector2, d_std);
    status = cudaGetLastError();
    if(status != cudaSuccess){printf("Error running the kernel");};
    ////////////////////////////////

    status = cudaMemcpy(h_cost, d_ncc, sizeof(float), cudaMemcpyDeviceToHost);
    if(status != cudaSuccess){printf("Error copying the cost function back the kernel");};

    status = cudaMemcpy(h_std, d_std, sizeof(float), cudaMemcpyDeviceToHost);
    if(status != cudaSuccess){printf("Error copying the std function back the kernel");};

    h_std[0] = sqrt(h_std[0]/(HEIGHT*WIDTH));
    h_cost[0] = -h_cost[0]/(HEIGHT*WIDTH*h_std[0]*h_xraystd);

    //status = cudaMemcpy(h_ncc, d_ncc, sizeof(float), cudaMemcpyDeviceToHost);
    //if(status != cudaSuccess){printf("Error copying the cost function back the kernel");};


    return status;
}


cudaError_t CudaEntryNGC1(float* h_Cam2CT, int* h_BoundingBox, float* h_rayParams, float* h_std, float h_xrayGxstd, float h_xrayGystd,
                          float* d_DRR, float* d_DRRGx, float* d_DRRGy, float* d_xrayGx, float* d_xrayGy,
                          float* d_vector1, float* d_vector2, float* d_mean, float* d_ncc, float* d_std, float* h_cost)
{
    cudaError_t status;

    status = cudaMemcpyToSymbol(consCam2CT1, h_Cam2CT, sizeof(float)*16);
    if(status != cudaSuccess){printf("Error copying transform to GPU");}
    status = cudaMemcpyToSymbol(consBoundingBox1, h_BoundingBox, sizeof(int)*4);
    if(status != cudaSuccess){printf("Error copying bounding box to GPU");}
    status = cudaMemcpyToSymbol(consRayParams1, h_rayParams, sizeof(float)*4);
    if(status != cudaSuccess){printf("Error copying ray params to GPU");}

    ////////////////////////////////
    int blkdim=16;
    int boxWidth = WIDTH;
    int boxHeight = HEIGHT;
    dim3 B1(boxWidth/blkdim,boxHeight/blkdim);
    dim3 T1(blkdim,blkdim);
    kernelRayCast1<<<B1,T1>>>(d_DRR);
    status = cudaGetLastError();
    if(status != cudaSuccess){printf("Error running the kernel");};
    ////////////////////////////////

    ////////////////////////////////

    //dim3 B2(boxHeight,boxWidth);
    //dim3 T2(blkdim,blkdim);

    status = CudaEntryFilterImage1(d_DRR, d_DRRGx, d_DRRGy);
    if(status != cudaSuccess){return status;};


    //////////////////////////////// GX
    KernelReduceImage1<<<HEIGHT,WIDTH>>>(d_DRRGx, d_vector1);
    KernelReduceImage1<<<1,HEIGHT>>>(d_vector1, d_mean);

    float costGx =0.0;
    KernelNCC1<<<HEIGHT,WIDTH>>>(d_DRRGx, d_xrayGx, d_mean, d_vector1, d_vector2);
    KernelReduceImage1<<<1,HEIGHT>>>(d_vector1, d_ncc);
    KernelReduceImage1<<<1,HEIGHT>>>(d_vector2, d_std);
    status = cudaGetLastError();
    if(status != cudaSuccess){printf("Error running the kernel Gx");};

    status = cudaMemcpy(h_cost, d_ncc, sizeof(float), cudaMemcpyDeviceToHost);
    if(status != cudaSuccess){printf("Error copying the cost function back the kernel");};

    status = cudaMemcpy(h_std, d_std, sizeof(float), cudaMemcpyDeviceToHost);
    if(status != cudaSuccess){printf("Error copying the std function back the kernel");};

    h_std[0] = sqrt(h_std[0]/(HEIGHT*WIDTH));
    costGx = -h_cost[0]/(HEIGHT*WIDTH*h_std[0]*h_xrayGxstd);
    /////////////////////////

    //////////////////////////////// GY
    KernelReduceImage1<<<HEIGHT,WIDTH>>>(d_DRRGy, d_vector1);
    KernelReduceImage1<<<1,HEIGHT>>>(d_vector1, d_mean);

    float costGy =0.0;
    KernelNCC1<<<HEIGHT,WIDTH>>>(d_DRRGy, d_xrayGy, d_mean, d_vector1, d_vector2);
    KernelReduceImage1<<<1,HEIGHT>>>(d_vector1, d_ncc);
    KernelReduceImage1<<<1,HEIGHT>>>(d_vector2, d_std);
    status = cudaGetLastError();
    if(status != cudaSuccess){printf("Error running the kernel Gx");};

    status = cudaMemcpy(h_cost, d_ncc, sizeof(float), cudaMemcpyDeviceToHost);
    if(status != cudaSuccess){printf("Error copying the cost function back the kernel");};

    status = cudaMemcpy(h_std, d_std, sizeof(float), cudaMemcpyDeviceToHost);
    if(status != cudaSuccess){printf("Error copying the std function back the kernel");};

    h_std[0] = sqrt(h_std[0]/(HEIGHT*WIDTH));
    costGy = -h_cost[0]/(HEIGHT*WIDTH*h_std[0]*h_xrayGystd);
    /////////////////////////

    h_cost[0] = 0.5*(costGx+costGy);
    //status = cudaMemcpy(h_ncc, d_ncc, sizeof(float), cudaMemcpyDeviceToHost);
    //if(status != cudaSuccess){printf("Error copying the cost function back the kernel");};


    return status;
}

cudaError_t CudaEntrySetConstantData1(float* camera, float xraymean)
{
	cudaError_t status;

    status = cudaMemcpyToSymbol(consCamera1, camera, sizeof(float)*12);
	if(status!=cudaSuccess){printf("Error copying camera parameters to GPU!\n");return status;}

    float invCamera[12];
    for( int k=0; k<12; k++)
    {
        invCamera[k]=1/camera[k];
    }
    status = cudaMemcpyToSymbol(consInvCamera1, invCamera, sizeof(float)*12);
    if(status!=cudaSuccess){printf("Error copying inverse camera parameters to GPU!\n");return status;}

    float xmean[1];
    xmean[0] = xraymean;

    status = cudaMemcpyToSymbol(consXrayMean1, xmean, sizeof(float));
    if(status!=cudaSuccess){printf("Error copying xray mean to GPU!\n");return status;}

	return status;
}

cudaError_t CudaEntrySetConstantData2(float* camera, float xraymean)
{
    cudaError_t status;

    status = cudaMemcpyToSymbol(consCamera2, camera, sizeof(float)*12);
    if(status!=cudaSuccess){printf("Error copying camera parameters to GPU!\n");return status;}

    float invCamera[12];
    for( int k=0; k<12; k++)
    {
        invCamera[k]=1/camera[k];
    }
    status = cudaMemcpyToSymbol(consInvCamera2, invCamera, sizeof(float)*12);
    if(status!=cudaSuccess){printf("Error copying inverse camera parameters to GPU!\n");return status;}

    float xmean[1];
    xmean[0] = xraymean;

    status = cudaMemcpyToSymbol(consXrayMean2, xmean, sizeof(float));
    if(status!=cudaSuccess){printf("Error copying xray mean to GPU!\n");return status;}

    return status;
}


cudaError_t CudaEntryBindCT(cudaArray* d_CT, cudaChannelFormatDesc chDescCT)
{

    printf("Binding texture to CT\n");
    cudaError_t status;
 	// set texture parameters
	texCT.normalized = false;
	texCT.filterMode = cudaFilterModeLinear;
	texCT.addressMode[0] = cudaAddressModeBorder;
	texCT.addressMode[1] = cudaAddressModeBorder;
	texCT.addressMode[2] = cudaAddressModeBorder;

  	// bind array to 3D texture
    status = cudaBindTextureToArray(texCT, d_CT, chDescCT);
	if(status!=cudaSuccess)
	{
		printf("Error binding CPU to texture!\n");
	}
	return status;
}

cudaError_t CudaEntryFilterImage1(float* d_img, float* d_Gx, float* d_Gy)
{
    cudaError_t status;

    FilterImageGx1<<<HEIGHT,WIDTH>>>(d_img,d_Gx);
    status = cudaGetLastError();
    if(status != cudaSuccess){printf("Error running the Gx kernel");};
    ////////////////////////////////

    FilterImageGy1<<<WIDTH,HEIGHT>>>(d_img,d_Gy);
    status = cudaGetLastError();
    if(status != cudaSuccess){printf("Error running the Gy kernel");};
    ////////////////////////////////

    return status;
}

cudaError_t CudaEntryFilterImage2(float* d_img, float* d_Gx, float* d_Gy)
{
    cudaError_t status;

    FilterImageGx2<<<HEIGHT,WIDTH>>>(d_img,d_Gx);
    status = cudaGetLastError();
    if(status != cudaSuccess){printf("Error running the Gx kernel");};
    ////////////////////////////////

    FilterImageGy2<<<WIDTH,HEIGHT>>>(d_img,d_Gy);
    status = cudaGetLastError();
    if(status != cudaSuccess){printf("Error running the Gy kernel");};
    ////////////////////////////////

    return status;
}

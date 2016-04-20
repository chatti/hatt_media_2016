#define TPB 512
#define pW 16
#define PATCHCOUNT 18

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


cudaError_t CudaEntryBindCT(cudaArray* d_CT, cudaChannelFormatDesc chDescCT);
cudaError_t EntrySetupTextureXrayGx1(cudaArray* d_xrayGx, cudaChannelFormatDesc chDesc);
cudaError_t EntrySetupTextureXrayGx2(cudaArray* d_xrayGy, cudaChannelFormatDesc chDesc);
cudaError_t EntrySetupTextureXrayGy1(cudaArray* d_xrayGx, cudaChannelFormatDesc chDesc);
cudaError_t EntrySetupTextureXrayGy2(cudaArray* d_xrayGy, cudaChannelFormatDesc chDesc);
cudaError_t CudaEntryPatchGCC(float* h_Cam2CT, float* h_rayParams, float* h_patchLocations, float* h_pGCC, float* d_pGCC);
cudaError_t CudaEntrySetConstantData1(float* camera);
cudaError_t CudaEntrySetConstantData2(float* camera);

__device__ void GeneratePatchGx(float* shrPatch, float* shrPatchG, int k);
__device__ void GeneratePatchGy(float* shrPatch, float* shrPatchG, int k);
__device__ void ReducePatch(float* shrPatch, float* shrReduce, int k);

/////////Textures//////////////////
texture<float, 3, cudaReadModeElementType> texCT;
texture<float, 2, cudaReadModeElementType> texXrayGx1;
texture<float, 2, cudaReadModeElementType> texXrayGy1;
texture<float, 2, cudaReadModeElementType> texXrayGx2;
texture<float, 2, cudaReadModeElementType> texXrayGy2;

/////////Constant data//////////////////
__constant__ float consCamera1[12];
__constant__ float consCamera2[12];
__constant__ float consInvCamera1[12];
__constant__ float consInvCamera2[12];
__constant__ float consRayParams1[4];
__constant__ float consRayParams2[4];
__constant__ float consInvCTparams[9];
__constant__ float consCam2CT1[16];
__constant__ float consCam2CT2[16];
__constant__ float consPatchLocations[2*PATCHCOUNT];


/*
__global__ void kernelPatchRayCast1(float* pGCC)
{


    __shared__ float shrPatchDRRGradient[pW*pW];
    __shared__ float shrPatchXrayGradient[pW*pW];
    __shared__ float shrScratchPad[pW*pW];

    __shared__ float shrDRRMean[1];
    __shared__ float shrDRRStd[1];
    __shared__ float shrXrayMean[1];
    __shared__ float shrXrayStd[1];

    int i = threadIdx.x;
    int j = threadIdx.y;

    float u   = consPatchLocations[blockIdx.x]             + (float)(i - pW/2);
    float v   = consPatchLocations[blockIdx.x +PATCHCOUNT] + (float)(j - pW/2);

    //k is the linear patch index
    int k = j*pW+i;

    //float SID    = consCamera1[4];       //SID
    float invSID = consInvCamera1[4];
    int steps  = (int)consRayParams1[2];
    //This are the locations of pixels, in physical coordinates
    //The origin is the center of the detector
    float x = consCamera1[2]*(u - 0.5*consCamera1[0]);
    float y = consCamera1[3]*(v - 0.5*consCamera1[1]);
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

    ///////////////////////////////////////////////////////////////////////////////////
    //Generate the DRR
    ///////////////////////////////////////////////////////////////////////////////////
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

    sum = sum/(sqrt(dx*dx+dy*dy+dz*dz)*steps);
    sum=sqrt(sum);

    shrScratchPad[k] = sum;
    __syncthreads();
    ///////////////////////////////////////////////////////////////////////////////////
    //Get the DRR and Xray gradients
    ///////////////////////////////////////////////////////////////////////////////////
    if(blockIdx.y==0)
    {
        GeneratePatchGx(shrScratchPad,shrPatchDRRGradient, k);
        shrPatchXrayGradient[k] = tex2D(texXrayGy1,u,v);
    }
    else
    {
        GeneratePatchGy(shrScratchPad,shrPatchDRRGradient, k);
        shrPatchXrayGradient[k] = tex2D(texXrayGx1,u,v);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    //Get the DRR gradient mean and std
    ///////////////////////////////////////////////////////////////////////////////////
    ReducePatch(shrPatchDRRGradient, shrScratchPad);
    if(i==0 & j==0)
    {
        shrDRRMean[0] = shrScratchPad[0]/(pW*pW);
    }
    __syncthreads();

    shrScratchPad[k] = shrPatchDRRGradient[k] - shrDRRMean[0];
    shrScratchPad[k] = shrScratchPad[k]*shrScratchPad[k];
    ReducePatch(shrScratchPad, shrScratchPad);
    if(i==0 & j==0)
    {
        shrDRRStd[0] = sqrt(shrScratchPad[0]);
    }
    __syncthreads();

    ///////////////////////////////////////////////////////////////////////////////////
    //Get the Xray gradient mean and std
    ///////////////////////////////////////////////////////////////////////////////////
    ReducePatch(shrPatchXrayGradient, shrScratchPad);
    if(i==0 & j==0)
    {
        shrXrayMean[0] = shrScratchPad[0]/(pW*pW);
    }
    __syncthreads();

    shrScratchPad[k] = shrPatchXrayGradient[k] - shrXrayMean[0];
    shrScratchPad[k] = shrScratchPad[k]*shrScratchPad[k];
    ReducePatch(shrScratchPad, shrScratchPad);
    if(i==0 & j==0)
    {
        shrXrayStd[0] = sqrt(shrScratchPad[0]);
    }
    __syncthreads();

    ///////////////////////////////////////////////////////////////////////////////////
    //Get the NCC between patches
    ///////////////////////////////////////////////////////////////////////////////////
    shrScratchPad[k] = (shrPatchXrayGradient[k] - shrXrayMean[0])*(shrPatchDRRGradient[k] - shrDRRMean[0]);

    ReducePatch(shrScratchPad, shrScratchPad);

    if(i==0 & j==0 & blockIdx.x==0)
    {
        printf("GCC M %d %f\n",blockIdx.y, shrScratchPad[0]/(shrDRRStd[0]*shrXrayStd[0]));
    }

    if(i==0 & j==0)
    {
        pGCC[blockIdx.x + PATCHCOUNT*blockIdx.y] = shrScratchPad[0]/(shrDRRStd[0]*shrXrayStd[0]);
    }

}*/


__global__ void kernelPatchRayCast1(float* pGCC)
{

    __shared__ float shrPatchDRRGradient[pW*pW];
    __shared__ float shrPatchXrayGradient[pW*pW];
    __shared__ float shrScratchPad[pW*pW];

    __shared__ float shrDRRMean[1];
    __shared__ float shrDRRStd[1];
    __shared__ float shrXrayMean[1];
    __shared__ float shrXrayStd[1];

    int i = threadIdx.x;
    int j = threadIdx.y;

    float u   = consPatchLocations[blockIdx.x]             + (float)(i - pW/2);
    float v   = consPatchLocations[blockIdx.x +PATCHCOUNT] + (float)(j - pW/2);

    //k is the linear patch index
    int k = j*pW+i;

    //float SID    = consCamera1[4];       //SID
    float invSID = consInvCamera1[4];
    int steps  = (int)consRayParams1[2];
    //This are the locations of pixels, in physical coordinates
    //The origin is the center of the detector
    float x = consCamera1[2]*(u+0.5 - 0.5*consCamera1[0]);
    float y = consCamera1[3]*(v+0.5 - 0.5*consCamera1[1]);
    //This is the initial, bottom most, z, in physical coordinates
    float z = consRayParams1[0];

    //compute the change in x or y per change in z
    //float dz = l/s;                              //deltaZ = consRayParams1[1]https://www.youtube.com/watch?v=-UBOfall9IY
    float dz = consRayParams1[3];
    float dx = dz*x*invSID;
    float dy = dz*y*invSID;

    //These are the initial x and y locations in the physical space
    x = x - z*x*invSID;
    y = y - z*y*invSID;

    ///////////////////////////////////////////////////////////////////////////////////
    //Generate the DRR
    ///////////////////////////////////////////////////////////////////////////////////
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
    sum = sum/(sqrt(dx*dx+dy*dy+dz*dz)*steps);
    sum=sqrt(sum);

    shrScratchPad[k] = sum;
    __syncthreads();
    ///////////////////////////////////////////////////////////////////////////////////
    //Get the DRR and Xray gradients
    ///////////////////////////////////////////////////////////////////////////////////
    if(blockIdx.y==0)
    {
        GeneratePatchGx(shrScratchPad,shrPatchDRRGradient,k);
        shrPatchXrayGradient[k] = tex2D(texXrayGy1,u,v);
    }
    else
    {
        GeneratePatchGy(shrScratchPad,shrPatchDRRGradient, k);
        shrPatchXrayGradient[k] = tex2D(texXrayGx1,u,v);
    }


    ///////////////////////////////////////////////////////////////////////////////////
    //Get the DRR gradient mean and std
    ///////////////////////////////////////////////////////////////////////////////////


    ReducePatch(shrPatchDRRGradient,shrScratchPad, k);
    if(i==0 & j==0)
    {
        shrDRRMean[0] = shrScratchPad[0]/(pW*pW);
    }
    __syncthreads();

    shrScratchPad[k] = shrPatchDRRGradient[k] - shrDRRMean[0];
    shrScratchPad[k] = shrScratchPad[k]*shrScratchPad[k];
    ReducePatch(shrScratchPad, shrScratchPad, k);
    if(i==0 & j==0)
    {
        shrDRRStd[0] = sqrt(shrScratchPad[0]);
    }
    __syncthreads();

    ///////////////////////////////////////////////////////////////////////////////////
    //Get the Xray gradient mean and std
    ///////////////////////////////////////////////////////////////////////////////////

    ReducePatch(shrPatchXrayGradient, shrScratchPad, k);
    if(i==0 & j==0)
    {
        shrXrayMean[0] = shrScratchPad[0]/(pW*pW);
    }
    __syncthreads();


    shrScratchPad[k] = shrPatchXrayGradient[k] - shrXrayMean[0];
    shrScratchPad[k] = shrScratchPad[k]*shrScratchPad[k];
    ReducePatch(shrScratchPad, shrScratchPad, k);
    if(i==0 & j==0)
    {
        shrXrayStd[0] = sqrt(shrScratchPad[0]);
    }

    __syncthreads();

    ///////////////////////////////////////////////////////////////////////////////////
    //Get the NCC between patches
    ///////////////////////////////////////////////////////////////////////////////////
    shrScratchPad[k] = (shrPatchXrayGradient[k] - shrXrayMean[0])*(shrPatchDRRGradient[k] - shrDRRMean[0]);
    ReducePatch(shrScratchPad, shrScratchPad, k);

    //if(i==0 & j==0 & blockIdx.x==0)
    //{
    //    printf("GCC M %d %f\n",blockIdx.y, shrScratchPad[0]/(shrDRRStd[0]*shrXrayStd[0]));
    //}

    if(i==0 & j==0)
    {
        //printf("%d %d %f\n", blockIdx.x, blockIdx.y, shrScratchPad[0]/(shrDRRStd[0]*shrXrayStd[0]));
        pGCC[blockIdx.x + PATCHCOUNT*blockIdx.y] = shrScratchPad[0]/(shrDRRStd[0]*shrXrayStd[0]);
    }

}

__global__ void kernelPatchRayCast1Test(float* patchesDRR, float* patchesIMG, float* pGCC)
{

    __shared__ float shrPatchDRRGradient[pW*pW];
    __shared__ float shrPatchXrayGradient[pW*pW];
    __shared__ float shrScratchPad[pW*pW];

    __shared__ float shrDRRMean[1];
    __shared__ float shrDRRStd[1];
    __shared__ float shrXrayMean[1];
    __shared__ float shrXrayStd[1];

    int i = threadIdx.x;
    int j = threadIdx.y;

    float u   = consPatchLocations[blockIdx.x]             + (float)(i - pW/2);
    float v   = consPatchLocations[blockIdx.x +PATCHCOUNT] + (float)(j - pW/2);

    //k is the linear patch index
    int k = j*pW+i;

    //float SID    = consCamera1[4];       //SID
    float invSID = consInvCamera1[4];
    int steps  = (int)consRayParams1[2];
    //This are the locations of pixels, in physical coordinates
    //The origin is the center of the detector
    float x = consCamera1[2]*(u+0.5 - 0.5*consCamera1[0]);
    float y = consCamera1[3]*(v+0.5 - 0.5*consCamera1[1]);
    //This is the initial, bottom most, z, in physical coordinates
    float z = consRayParams1[0];

    //compute the change in x or y per change in z
    //float dz = l/s;                              //deltaZ = consRayParams1[1]https://www.youtube.com/watch?v=-UBOfall9IY
    float dz = consRayParams1[3];
    float dx = dz*x*invSID;
    float dy = dz*y*invSID;

    //These are the initial x and y locations in the physical space
    x = x - z*x*invSID;
    y = y - z*y*invSID;

    ///////////////////////////////////////////////////////////////////////////////////
    //Generate the DRR
    ///////////////////////////////////////////////////////////////////////////////////
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
    sum = sum/(sqrt(dx*dx+dy*dy+dz*dz)*steps);
    sum=sqrt(sum);

    shrScratchPad[k] = sum;
    __syncthreads();
    ///////////////////////////////////////////////////////////////////////////////////
    //Get the DRR and Xray gradients
    ///////////////////////////////////////////////////////////////////////////////////
    if(blockIdx.y==0)
    {
        GeneratePatchGx(shrScratchPad,shrPatchDRRGradient,k);
        shrPatchXrayGradient[k] = tex2D(texXrayGy1,u,v);
    }
    else
    {
        GeneratePatchGy(shrScratchPad,shrPatchDRRGradient, k);
        shrPatchXrayGradient[k] = tex2D(texXrayGx1,u,v);
    }


    //patchesIMG[blockIdx.y*32*32*24 + blockIdx.x*32*32 + threadIdx.y*32 + threadIdx.x] = shrPatchXrayGradient[k];
    //patchesDRR[blockIdx.y*32*32*24 + blockIdx.x*32*32 + threadIdx.y*32 + threadIdx.x] = shrPatchDRRGradient[k];

    patchesDRR[blockIdx.y*pW*pW*PATCHCOUNT + blockIdx.x*pW*pW + threadIdx.y*pW + threadIdx.x] = shrPatchDRRGradient[k];
    patchesIMG[blockIdx.y*pW*pW*PATCHCOUNT + blockIdx.x*pW*pW + threadIdx.y*pW + threadIdx.x] = shrPatchXrayGradient[k];
    ///////////////////////////////////////////////////////////////////////////////////
    //Get the DRR gradient mean and std
    ///////////////////////////////////////////////////////////////////////////////////
    //if(i==0 & j==0 & blockIdx.x==0)
    //{
    //    printf("DRR    %d %f\n",blockIdx.y, shrScratchPad[32*15 + 15]);
    //    printf("DRRGx  %d %f\n",blockIdx.y, shrPatchDRRGradient[32*15 + 15]);
    //}

    ReducePatch(shrPatchDRRGradient,shrScratchPad, k);
    if(i==0 & j==0)
    {
        shrDRRMean[0] = shrScratchPad[0]/(pW*pW);
    }
    __syncthreads();

    //if(i==0 & j==0 & blockIdx.x==0)
    //{
    //    printf("DRRGxM %d %f\n",blockIdx.y, shrDRRMean[0]);
    //}

    shrScratchPad[k] = shrPatchDRRGradient[k] - shrDRRMean[0];
    shrScratchPad[k] = shrScratchPad[k]*shrScratchPad[k];
    ReducePatch(shrScratchPad, shrScratchPad, k);
    if(i==0 & j==0)
    {
        shrDRRStd[0] = sqrt(shrScratchPad[0]);
    }
    __syncthreads();

    //if(i==0 & j==0 & blockIdx.x==0)
    //{
    //    printf("DRRGxS %d %f\n",blockIdx.y, shrDRRStd[0]);
    //}

    ///////////////////////////////////////////////////////////////////////////////////
    //Get the Xray gradient mean and std
    ///////////////////////////////////////////////////////////////////////////////////

    if(i==0 & j==0 & blockIdx.x==0)
    {
        //printf("IMGG   %d %f\n",blockIdx.y, shrPatchXrayGradient[32*15 + 15]);
    }

    ReducePatch(shrPatchXrayGradient, shrScratchPad, k);
    if(i==0 & j==0)
    {
        shrXrayMean[0] = shrScratchPad[0]/(pW*pW);
    }
    __syncthreads();

    if(i==0 & j==0 & blockIdx.x==0)
    {
        //printf("IMGG M %d %f\n",blockIdx.y, shrXrayMean[0]);
    }

    shrScratchPad[k] = shrPatchXrayGradient[k] - shrXrayMean[0];
    shrScratchPad[k] = shrScratchPad[k]*shrScratchPad[k];
    ReducePatch(shrScratchPad, shrScratchPad, k);
    if(i==0 & j==0)
    {
        shrXrayStd[0] = sqrt(shrScratchPad[0]);
    }

    //if(i==0 & j==0 & blockIdx.x==0)
    //{
    //    printf("IMGG M %d %f\n",blockIdx.y, shrXrayStd[0]);
    //}

    __syncthreads();

    ///////////////////////////////////////////////////////////////////////////////////
    //Get the NCC between patches
    ///////////////////////////////////////////////////////////////////////////////////
    shrScratchPad[k] = (shrPatchXrayGradient[k] - shrXrayMean[0])*(shrPatchDRRGradient[k] - shrDRRMean[0]);

    ReducePatch(shrScratchPad, shrScratchPad, k);

    //if(i==0 & j==0 & blockIdx.x==0)
    //{
    //    printf("GCC M %d %f\n",blockIdx.y, shrScratchPad[0]/(shrDRRStd[0]*shrXrayStd[0]));
    //}

    if(i==0 & j==0)
    {
        //printf("%d %d %f\n", blockIdx.x, blockIdx.y, shrScratchPad[0]/(shrDRRStd[0]*shrXrayStd[0]));
        pGCC[blockIdx.x + PATCHCOUNT*blockIdx.y] = shrScratchPad[0]/(shrDRRStd[0]*shrXrayStd[0]);
    }



}

__device__ void GeneratePatchGx(float* shrPatch, float* shrPatchG, int k)
{

    if(threadIdx.x>=1 & threadIdx.x<(pW-1))
    {
        shrPatchG[k] = 0.5*shrPatch[k-1] - 0.5*shrPatch[k+1];
    }
    else
    {
        shrPatchG[k]=0.0;
    }
}

__device__ void GeneratePatchGy(float* shrPatch, float* shrPatchG, int k)
{

    if(threadIdx.y>=1 & threadIdx.y<(pW-1))
    {
        shrPatchG[k] = 0.5*shrPatch[k-pW] - 0.5*shrPatch[k+pW];
    }
    else
    {
        shrPatchG[k]=0.0;
    }
}

__device__ void ReducePatch(float* shrPatch, float* shrReduce, int k)
{
    shrReduce[k] = shrPatch[k]; __syncthreads();


    #if pW == 32
    if (k<512) { shrReduce[k] += shrReduce[k+512];   } __syncthreads();
    if (k<256) { shrReduce[k] += shrReduce[k+256];   } __syncthreads();
    #endif

    if (k<128) { shrReduce[k] += shrReduce[k+128];   } __syncthreads();
    if (k<64 ) { shrReduce[k] += shrReduce[k+64 ];   } __syncthreads();
    if (k<32 ) { shrReduce[k] += shrReduce[k+32 ];   } __syncthreads();
    if (k<16 ) { shrReduce[k] += shrReduce[k+16 ];   } __syncthreads();
    if (k<8  ) { shrReduce[k] += shrReduce[k+8  ];   } __syncthreads();
    if (k<4  ) { shrReduce[k] += shrReduce[k+4  ];   } __syncthreads();
    if (k<2  ) { shrReduce[k] += shrReduce[k+2  ];   } __syncthreads();
    if (k<1  ) { shrReduce[k] += shrReduce[k+1  ];   } __syncthreads();

}

cudaError_t CudaEntryPatchGCCTest(float* h_Cam2CT, float* h_rayParams, float* h_patchLocations, float* h_pGCC, float* d_pGCC)
{
    cudaError_t status;

    status = cudaMemcpyToSymbol(consCam2CT1, h_Cam2CT, sizeof(float)*16);
    if(status != cudaSuccess){printf("Error copying transform to GPU");}
    status = cudaMemcpyToSymbol(consPatchLocations, h_patchLocations, sizeof(float)*PATCHCOUNT*2);
    if(status != cudaSuccess){printf("Error copying patch locations box to GPU");}
    status = cudaMemcpyToSymbol(consRayParams1, h_rayParams, sizeof(float)*4);
    if(status != cudaSuccess){printf("Error copying ray params to GPU");}

    float hPatchesDRR[pW*pW*PATCHCOUNT*2];
    float hPatchesIMG[pW*pW*PATCHCOUNT*2];
    float* dPatchesDRR;
    float* dPatchesIMG;
    status = cudaMalloc((void **) &dPatchesDRR,   pW*pW*PATCHCOUNT*2*sizeof(float));
    status = cudaMalloc((void **) &dPatchesIMG,   pW*pW*PATCHCOUNT*2*sizeof(float));
    ////////////////////////////////
    dim3 B(PATCHCOUNT,2);
    dim3 T(pW,pW);
    kernelPatchRayCast1Test<<<B,T>>>(dPatchesDRR,dPatchesIMG, d_pGCC);
    status = cudaGetLastError();
    if(status != cudaSuccess){printf("Error running the patch ray casting kernel");};
    ////////////////////////////////

    status = cudaMemcpy(hPatchesDRR, dPatchesDRR, pW*pW*PATCHCOUNT*2*sizeof(float), cudaMemcpyDeviceToHost);
    status = cudaMemcpy(hPatchesIMG, dPatchesIMG, pW*pW*PATCHCOUNT*2*sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess){printf("Copying cost back to the host failed!\n");return status;}

    FILE *fp;

    fp = fopen("patchesDRR.raw", "w");
    if(fp==NULL){
        printf("Error writing: \n");
    }
    size_t r = fwrite(hPatchesDRR, sizeof(float), pW*pW*PATCHCOUNT*2, fp);
    fclose(fp);

    fp = fopen("patchesIMG.raw", "w");
    if(fp==NULL){
        printf("Error writing: \n");
    }
    r = fwrite(hPatchesIMG, sizeof(float), pW*pW*PATCHCOUNT*2, fp);
    fclose(fp);



    return status;
}


cudaError_t CudaEntryPatchGCC(float* h_Cam2CT, float* h_rayParams, float* h_patchLocations, float* h_pGCC, float* d_pGCC)
{
    cudaError_t status;

    status = cudaMemcpyToSymbol(consCam2CT1, h_Cam2CT, sizeof(float)*16);
    if(status != cudaSuccess){printf("Error copying transform to GPU");}
    status = cudaMemcpyToSymbol(consPatchLocations, h_patchLocations, sizeof(float)*PATCHCOUNT*2);
    if(status != cudaSuccess){printf("Error copying patch locations box to GPU");}
    status = cudaMemcpyToSymbol(consRayParams1, h_rayParams, sizeof(float)*4);
    if(status != cudaSuccess){printf("Error copying ray params to GPU");}

    ////////////////////////////////
    dim3 B(PATCHCOUNT,2);
    dim3 T(pW,pW);
    kernelPatchRayCast1<<<B,T>>>(d_pGCC);
    status = cudaGetLastError();
    if(status != cudaSuccess){printf("Error running the patch ray casting kernel");};
    ////////////////////////////////

    status = cudaMemcpy(h_pGCC, d_pGCC, 2*PATCHCOUNT*sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess){printf("Copying cost back to the host failed!\n");return status;}


    return status;
}


cudaError_t EntrySetupTextureXrayGx1(cudaArray* d_xray, cudaChannelFormatDesc chDesc)
{
    cudaError_t status;

    texXrayGx1.addressMode[0] = cudaAddressModeBorder;
    texXrayGx1.addressMode[1] = cudaAddressModeBorder;
    texXrayGx1.filterMode     = cudaFilterModeLinear;
    texXrayGx1.normalized     = false;

    status = cudaBindTextureToArray(texXrayGx1, d_xray,  chDesc);
    if (status != cudaSuccess){printf("Binding the texture failed!\n");return status;}
    return status;
}

cudaError_t EntrySetupTextureXrayGx2(cudaArray* d_xray, cudaChannelFormatDesc chDesc)
{
    cudaError_t status;

    texXrayGx2.addressMode[0] = cudaAddressModeBorder;
    texXrayGx2.addressMode[1] = cudaAddressModeBorder;
    texXrayGx2.filterMode     = cudaFilterModeLinear;
    texXrayGx2.normalized     = false;

    status = cudaBindTextureToArray(texXrayGx2, d_xray,  chDesc);
    if (status != cudaSuccess){printf("Binding the texture failed!\n");return status;}
    return status;
}

cudaError_t EntrySetupTextureXrayGy1(cudaArray* d_xray, cudaChannelFormatDesc chDesc)
{
    cudaError_t status;

    texXrayGy1.addressMode[0] = cudaAddressModeBorder;
    texXrayGy1.addressMode[1] = cudaAddressModeBorder;
    texXrayGy1.filterMode     = cudaFilterModeLinear;
    texXrayGy1.normalized     = false;

    status = cudaBindTextureToArray(texXrayGy1, d_xray,  chDesc);
    if (status != cudaSuccess){printf("Binding the texture failed!\n");return status;}
    return status;
}

cudaError_t EntrySetupTextureXrayGy2(cudaArray* d_xray, cudaChannelFormatDesc chDesc)
{
    cudaError_t status;

    texXrayGy2.addressMode[0] = cudaAddressModeBorder;
    texXrayGy2.addressMode[1] = cudaAddressModeBorder;
    texXrayGy2.filterMode     = cudaFilterModeLinear;
    texXrayGy2.normalized     = false;

    status = cudaBindTextureToArray(texXrayGy2, d_xray,  chDesc);
    if (status != cudaSuccess){printf("Binding the texture failed!\n");return status;}
    return status;
}

cudaError_t CudaEntrySetConstantData1(float* camera)
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

    return status;
}

cudaError_t CudaEntrySetConstantData2(float* camera)
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



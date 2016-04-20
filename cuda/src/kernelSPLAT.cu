#define WIDTH 512
#define HEIGHT 512
#define bins 128

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
__constant__ float d_Scratch[1];
__constant__ float d_DRRMean[1];
__constant__ float d_DRRGMean[1];
__constant__ float d_XRAY1Mean[1];
__constant__ float d_XRAY1GMean[1];
__constant__ float d_XRAY2Mean[1];
__constant__ float d_XRAY2GMean[1];


/////////////////////////////////////////

__global__ void FilterImageGy(float* dimg, float* dGy)
{
    __shared__ float shrCol[HEIGHT];
    int w = WIDTH;
    int h = HEIGHT;
    int i = threadIdx.x;
    int j = w*i + blockIdx.x;

    shrCol[i] = dimg[j];
    __syncthreads();
    if(i>=2 & i < h-2)
    {
        shrCol[i] = 0.0545*shrCol[i-2]+0.2442*shrCol[i-1]+0.4026*shrCol[i]+0.2442*shrCol[i+1]+0.0545*shrCol[i+2];
        __syncthreads();
        dGy[j] = -0.5*shrCol[i-1]+0.5*shrCol[i+1];
    }
    else
    {
        dGy[j]=0.0;
    }

}

__global__ void FilterImageGx(float* dimg, float* dGx)
{
    __shared__ float shrRow[WIDTH];
    int w = WIDTH;
    int i = threadIdx.x;
    int j = w*blockIdx.x + i;


    shrRow[i] = dimg[j];
    __syncthreads();
    if(i>=2 & i < w-2)
    {
        shrRow[i] = 0.0545*shrRow[i-2]+0.2442*shrRow[i-1]+0.4026*shrRow[i]+0.2442*shrRow[i+1]+0.0545*shrRow[i+2];
        __syncthreads();
        dGx[j] = -0.5*shrRow[i-1]+0.5*shrRow[i+1];
    }
    else
    {
        dGx[j]=0.0;
    }

}



__global__ void ComputeMeanKernel(float* img, float * mean)
{
    __shared__ float shrMean[WIDTH];

    int i = threadIdx.x;
    int j = blockIdx.x;

    int gidx = ((int)d_camera1[0])*j + i;
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
    if (i<1)   { mean[j] = shrMean[i] + shrMean[i+1]; }

}

__global__ void ComputeStdKernel(float* img, float * std)
{
    __shared__ float shrStd[WIDTH];

    int i = threadIdx.x;
    int j = blockIdx.x;

    int gidx = ((int)d_camera1[0])*j + i;
    float val=0.0;
    val = img[gidx]-d_Scratch[0];
    shrStd[i] = val*val;
    __syncthreads();

    //Reduce the value
    if (i<256) { shrStd[i] += shrStd[i+256]; } __syncthreads();
    if (i<128) { shrStd[i] += shrStd[i+128]; } __syncthreads();
    if (i<64)  { shrStd[i] += shrStd[i+64];  } __syncthreads();
    if (i<32)  { shrStd[i] += shrStd[i+32];  } __syncthreads();
    if (i<16)  { shrStd[i] += shrStd[i+16];  } __syncthreads();
    if (i<8)   { shrStd[i] += shrStd[i+8];   } __syncthreads();
    if (i<4)   { shrStd[i] += shrStd[i+4];   } __syncthreads();
    if (i<2)   { shrStd[i] += shrStd[i+2];   } __syncthreads();
    if (i<1)   { std[j] = shrStd[i] + shrStd[i+1]; }

}

__global__ void NGC1_Kernel(float* drr, float* xray, float* cost, float * std)
{
    __shared__ float shrCost[WIDTH];
    __shared__ float shrStd[WIDTH];

    int i = threadIdx.x;
    int j = blockIdx.x;

    int gidx = ((int)d_camera1[0])*j + i;
    float val = drr[gidx] - d_DRRGMean[0];
    shrStd[i] = val*val;
    shrCost[i] = val*(xray[gidx] - d_XRAY1GMean[0]);



    __syncthreads();
    //Reduce the value
    if (i<256) { shrCost[i] += shrCost[i+256];shrStd[i] += shrStd[i+256]; } __syncthreads();
    if (i<128) { shrCost[i] += shrCost[i+128];shrStd[i] += shrStd[i+128]; } __syncthreads();
    if (i<64)  { shrCost[i] += shrCost[i+64]; shrStd[i] += shrStd[i+64];  } __syncthreads();
    if (i<32)  { shrCost[i] += shrCost[i+32]; shrStd[i] += shrStd[i+32];  } __syncthreads();
    if (i<16)  { shrCost[i] += shrCost[i+16]; shrStd[i] += shrStd[i+16];  } __syncthreads();
    if (i<8)   { shrCost[i] += shrCost[i+8];  shrStd[i] += shrStd[i+8];   } __syncthreads();
    if (i<4)   { shrCost[i] += shrCost[i+4];  shrStd[i] += shrStd[i+4];   } __syncthreads();
    if (i<2)   { shrCost[i] += shrCost[i+2];  shrStd[i] += shrStd[i+2];   } __syncthreads();
    if (i<1)   { cost[j] = shrCost[i]+shrCost[i+1]; std[j] = shrStd[i]+shrStd[i+1]; }

}

__global__ void NCC1_Kernel(float* drr, float* xray, float* cost, float * std)
{
    __shared__ float shrCost[WIDTH];
    __shared__ float shrStd[WIDTH];

    int i = threadIdx.x;
    int j = blockIdx.x;

    int gidx = ((int)d_camera1[0])*j + i;
    float val = drr[gidx] - d_DRRMean[0];
    shrStd[i] = val*val;
    shrCost[i] = val*(xray[gidx] - d_XRAY1Mean[0]);



    __syncthreads();
    //Reduce the value
    if (i<256) { shrCost[i] += shrCost[i+256];shrStd[i] += shrStd[i+256]; } __syncthreads();
    if (i<128) { shrCost[i] += shrCost[i+128];shrStd[i] += shrStd[i+128]; } __syncthreads();
    if (i<64)  { shrCost[i] += shrCost[i+64]; shrStd[i] += shrStd[i+64];  } __syncthreads();
    if (i<32)  { shrCost[i] += shrCost[i+32]; shrStd[i] += shrStd[i+32];  } __syncthreads();
    if (i<16)  { shrCost[i] += shrCost[i+16]; shrStd[i] += shrStd[i+16];  } __syncthreads();
    if (i<8)   { shrCost[i] += shrCost[i+8];  shrStd[i] += shrStd[i+8];   } __syncthreads();
    if (i<4)   { shrCost[i] += shrCost[i+4];  shrStd[i] += shrStd[i+4];   } __syncthreads();
    if (i<2)   { shrCost[i] += shrCost[i+2];  shrStd[i] += shrStd[i+2];   } __syncthreads();
    if (i<1)   { cost[j] = shrCost[i]+shrCost[i+1]; std[j] = shrStd[i]+shrStd[i+1]; }

}

__global__ void NCC2_Kernel(float* drr, float* xray, float* cost, float * std)
{
    __shared__ float shrCost[WIDTH];
    __shared__ float shrStd[WIDTH];

    int i = threadIdx.x;
    int j = blockIdx.x;
    int gidx = ((int)d_camera2[0])*j + i;
    float val = drr[gidx] - d_DRRMean[0];
    shrStd[i] = val*val;
    shrCost[i] = val*(xray[gidx] - d_XRAY2Mean[0]);


    __syncthreads();
    //Reduce the value
    if (i<256) { shrCost[i] += shrCost[i+256];shrStd[i] += shrStd[i+256]; } __syncthreads();
    if (i<128) { shrCost[i] += shrCost[i+128];shrStd[i] += shrStd[i+128]; } __syncthreads();
    if (i<64)  { shrCost[i] += shrCost[i+64]; shrStd[i] += shrStd[i+64];  } __syncthreads();
    if (i<32)  { shrCost[i] += shrCost[i+32]; shrStd[i] += shrStd[i+32];  } __syncthreads();
    if (i<16)  { shrCost[i] += shrCost[i+16]; shrStd[i] += shrStd[i+16];  } __syncthreads();
    if (i<8)   { shrCost[i] += shrCost[i+8];  shrStd[i] += shrStd[i+8];   } __syncthreads();
    if (i<4)   { shrCost[i] += shrCost[i+4];  shrStd[i] += shrStd[i+4];   } __syncthreads();
    if (i<2)   { shrCost[i] += shrCost[i+2];  shrStd[i] += shrStd[i+2];   } __syncthreads();
    if (i<1)   { cost[j] = shrCost[i]+shrCost[i+1]; std[j] = shrStd[i]+shrStd[i+1]; }

}

__global__ void SPLAT_Kernel(float* PC, float* T, int* splatIdx)
{

	int i = threadIdx.x;
    int j = blockIdx.x*blockDim.x + i; 


	float x   = PC[j];
    float y   = PC[j+d_pcParams[0]];
    float z   = PC[j+2*d_pcParams[0]];

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
    y = d_camera1[1] - (y+d_camera1[1]*0.5);
    //y = y+d_camera1[1]*0.5;
	
    int u = (int)(x);
    int v = (int)(y);

    splatIdx[j] = v*(int)d_camera1[0]+u;

}

cudaError_t EntryComputeGradient(float* dimg, float* dGx, float* dGy)
{
    cudaError_t status;
    FilterImageGx<<<HEIGHT,WIDTH>>>(dimg,dGx);
    status = cudaGetLastError();
    if (status != cudaSuccess){printf("Problem with x gradient computation kernel!\n");return status;}
    FilterImageGy<<<WIDTH,HEIGHT>>>(dimg,dGy);
    status = cudaGetLastError();
    if (status != cudaSuccess){printf("Problem with y gradient computation kernel!\n");return status;}

    return status;

}


cudaError_t EntryNCCSPLAT_Mono(float* h_drr, float* d_drr, int* h_splatIdx, int* d_splatIdx,
                          float* h_pcloud, float* d_pcloud, float* d_cost,
                          float* d_std, float* h_xray, float* d_xray, float* d_M, float* h_camera, int N, float xraystd, float xraymean, float drrmean, float &h_finalCost)
{

    //float h_std[HEIGHT];
    //float h_cost[HEIGHT];
    int idx=0;
    int T=WIDTH;
    int B=N/T;
    int pxcount = WIDTH*HEIGHT;
    //float fpxcount = h_camera[0]*h_camera[1];
    cudaError_t status;

    SPLAT_Kernel<<<B,T>>>(d_pcloud, d_M, d_splatIdx);
	status = cudaGetLastError();
    if (status != cudaSuccess){printf("Problem SPLAT kernel!\n");return status;}
	
    status = cudaMemcpy(h_splatIdx, d_splatIdx, N*sizeof(int), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess){printf("Copying cost back to the host failed!\n");return status;}

    int minx=0;
    int maxx=511;
    int miny=0;
    int maxy=511;

    //int curx=0;
    //int cury=0;

    float sumsum=0;
    memset(h_drr, 0, pxcount*sizeof(float) );

    int minidx = 512*512 - 1;
    int maxidx = 0;
    for(int k=0; k<N; k++)
	{
        idx = h_splatIdx[k];

        if(idx >=0 & idx<pxcount)
        {
            h_drr[idx]+=h_pcloud[3*N+k];
            sumsum+=h_pcloud[3*N+k];
            //printf("h_drr val %f idx %d\n", h_drr[idx],idx);
            if(idx < minidx)
            {
                minidx=idx;
            }
            else if ( idx > maxidx)
            {
                maxidx  = idx;
            }
        }

        //cury = (idx+1)/WIDTH-1;
        //curx = (idx+1)%WIDTH-1;
        //if(idx >=0 & idx<pxcount)
        //{
        //    h_drr[idx]+=h_pcloud[3*N+k];
        //    printf("idx %d val %f\n",idx,h_drr[idx]);
        //}
        /*if(curx < minx)
        {
            minx = curx;
        }
        else if(curx > maxx)
        {
            maxx = curx;
        }

        if(cury < miny)
        {
            miny = cury;
        }
        else if(cury > maxy)
        {
            maxy = cury;
        }
        */
	}

    //printf("minidx %d maxidx %d\n",minidx, maxidx);
    /*
    for( int x=minx; x<maxx; x++)
    {
        for( int y=miny; y <maxy; y++)
        {
            lidx = cury*WIDTH + curx;
            drrsum =drrsum + h_drr[lidx];
            xraysum = xraysum + h_xray[lidx];
            cnt++;
        }
    }

    float drrmean = drrsum/cnt;
    float xraymean = xraysum/cnt;
    */

    int lidx=0;
    float drrsum=0.0;
    float xraysum=0.0;
    float cnt=0.0;
    float ncc=0.0;
    float curdrrval = 0.0;
    float curxrayval = 0.0;
    for( int x=minx; x<maxx; x++)
    {
        for( int y=miny; y <maxy; y++)
        {
            lidx = y*WIDTH + x;
            curdrrval = (h_drr[lidx]-drrmean);
            curxrayval = (h_xray[lidx]-xraymean);
            ncc = ncc + (curdrrval*curxrayval);
            drrsum =drrsum + curdrrval*curdrrval;
            xraysum = xraysum + curxrayval*curxrayval;
            cnt++;

        }
    }

    float drrstd = sqrt(drrsum/cnt);

    //float nccpre = ncc;
    ncc=ncc/(cnt*drrstd*xraystd);
    h_finalCost = -ncc;

    //printf("minx %d, maxx %d, miny %d, maxy %d meandrr %f meanxray %f, drrstd %f, xraystd %f, ncc %f, nccpre %f \n", minx, maxx, miny, maxy, drrmean, xraymean,drrstd,xraystd, ncc,nccpre);

    return status;
    /*status = cudaMemcpy(d_drr, h_drr, pxcount*sizeof(float), cudaMemcpyHostToDevice);
    if (status != cudaSuccess){printf("Copying splat drr to the device failed!\n");return status;}

    NCC1_Kernel<<<HEIGHT,WIDTH>>>(d_drr, d_xray, d_cost, d_std);
    status = cudaGetLastError();
    if (status != cudaSuccess){printf("Problem computing ncc cost function!\n");return status;}

    status = cudaMemcpy(h_cost, d_cost, HEIGHT*sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess){printf("Copying cost to the host failed!\n");return status;}
    status = cudaMemcpy(h_std, d_std, HEIGHT*sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess){printf("Copying std vals to the host failed!\n");return status;}

    float drrstd = 0.0;
    float ncc = 0.0;
    for( int k=0; k<HEIGHT; k++)
    {
        drrstd+=h_std[k];
        ncc   +=h_cost[k];
    }
    drrstd=sqrt(drrstd/fpxcount);
    ncc=ncc/fpxcount;
    ncc=ncc/(drrstd*xraystd);
    h_finalCost = 1-ncc;

    //printf("ncc %f \n",ncc);
    //printf("drrstd %f \n",drrstd);
    //printf("xrystd %f \n",xraystd);

	return status;
    */
}

cudaError_t EntryMISPLAT_Mono(float* h_drr, int* h_splatIdx, int* d_splatIdx,
                              float* h_pcloud, float* d_pcloud, float* h_xray,
                              float* d_M, int N, float xraymax, float &h_finalCost)
{


    float PJoint[bins*bins]={0};
    float PDRR[bins]={0};
    float PXray[bins]={0};


    int idx=0;
    int T=WIDTH;
    int B=N/T;
    int pxcount = WIDTH*HEIGHT;
    cudaError_t status;

    SPLAT_Kernel<<<B,T>>>(d_pcloud, d_M, d_splatIdx);
    status = cudaGetLastError();
    if (status != cudaSuccess){printf("Problem SPLAT kernel!\n");return status;}

    status = cudaMemcpy(h_splatIdx, d_splatIdx, N*sizeof(int), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess){printf("Copying cost back to the host failed!\n");return status;}

    float drrmax=0.0;
    memset(h_drr, 0, pxcount*sizeof(float) );
    for(int k=0; k<N; k++)
    {
        idx = h_splatIdx[k];
        if(idx >=0 & idx<pxcount)
        {
            h_drr[idx]+=h_pcloud[3*N+k];
            if(h_drr[idx] > drrmax)
            {
                drrmax=h_drr[idx];
            }
        }
    }

    //memset(PJoint, 0, pxcount*sizeof(int) );
    //memset(PDRR,   0, pxcount*sizeof(int) );
    //memset(PXray,  0, pxcount*sizeof(int) );

    int dval=0;
    int xval=0;
    float fpxcount = (float)pxcount;
    for(int k=0; k <pxcount; k++)
    {
        if(h_drr[k]>0)
        {
            dval = (int)(bins * (h_drr[k]/drrmax));
            xval = (int)(bins * (h_xray[k]/xraymax));
            if(dval<0){dval=0;}
            if(xval<0){xval=0;}
            if(dval>(bins-1)){dval=bins-1;}
            if(xval>(bins-1)){xval=bins-1;}
            PDRR[dval] = PDRR[dval] + 1/fpxcount;
            PXray[xval] = PXray[xval] + 1/fpxcount;
            PJoint[bins*dval + xval] = PJoint[bins*dval + xval] + 1/fpxcount;
        }
    }

    float MI = 0.0;
    /////////debugging//////////
    /*
    FILE *fp;
    fp = fopen("PXY.raw", "w");
    size_t r = fwrite(PJoint, sizeof(float), 128*128, fp);
    fclose(fp);

    fp = fopen("PX.raw", "w");
    r = fwrite(PDRR, sizeof(float), 128, fp);
    fclose(fp);

    fp = fopen("PY.raw", "w");
    r = fwrite(PXray, sizeof(float), 128, fp);
    fclose(fp);

    fp = fopen("xray.raw", "w");
    r = fwrite(h_xray, sizeof(float), 512*512, fp);
    fclose(fp);

    fp = fopen("drr.raw", "w");
    r = fwrite(h_drr, sizeof(float), 512*512, fp);
    fclose(fp);
*/
    //////////
    for(int i=0; i<bins; i++)
    {
        if(PDRR[i]>0)
        {
            for(int j=0; j<bins; j++)
            {
                if(PXray[j]>0)
                {
                    if(PJoint[i*bins+j] > 0)
                    {
                        //printf("PXY %f \n",PJoint[i*bins+j]);
                        //printf("PX %f \n",PDRR[i]);
                        //printf("PY %f \n",PXray[j]);
                        //printf("log %f \n",log2(PJoint[i*bins+j]/(PDRR[i]*PXray[j]) ));

                        MI+=PJoint[i*bins+j]*log2(PJoint[i*bins+j]/(PDRR[i]*PXray[j]));
                    }
                }
            }
        }
    }


    h_finalCost = -MI;

    //printf("mi %f drrmax %f xraymax %f\n",MI, drrmax, xraymax);

    //printf("drrstd %f \n",drrstd);
    //printf("xrystd %f \n",xraystd);

    return status;

}

cudaError_t ComputeImageMean(float &mean, float* dimg, float* dmeancol)
{
    float hmeancol[WIDTH];
    cudaError_t status;
    ComputeMeanKernel<<<HEIGHT,WIDTH>>>(dimg, dmeancol);
    status = cudaGetLastError();
    if (status != cudaSuccess){printf("Problem mean computation kernel!\n");return status;}
    status = cudaMemcpy(hmeancol, dmeancol, HEIGHT*sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess){printf("Problem memcpy mean!\n");return status;}
    float pxcount = (float)(WIDTH*HEIGHT);
    mean = 0.0;
    for(int k=0; k<HEIGHT;k++)
    {
        mean+=hmeancol[k];
    }
    mean=mean/pxcount;
    return status;

}

cudaError_t ComputeImageStd(float &std, float mean, float* dimg, float* dstdcol)
{
    cudaError_t status;
    float mn[1];mn[0] = mean;
    float hstdcol[HEIGHT];
    status = cudaMemcpyToSymbol(d_Scratch, mn, sizeof(float));
    if(status!=cudaSuccess){printf("Error copying constant mean to GPU!\n");return status;}

    ComputeStdKernel<<<HEIGHT,WIDTH>>>(dimg, dstdcol);
    status = cudaGetLastError();
    if (status != cudaSuccess){printf("Error std computation kernel!\n");return status;}
    status = cudaMemcpy(hstdcol, dstdcol, HEIGHT*sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess){printf("Error memcpy std!\n");return status;}
    float pxcount = (float)(WIDTH*HEIGHT);
    std = 0.0;
    for(int k=0; k<HEIGHT;k++)
    {
        std+=hstdcol[k];
    }
    std=sqrt(std/pxcount);
    return status;

}

cudaError_t EntryNGCSPLAT_Mono(float* h_drr, float* d_drr, float* d_drrGx, float* d_drrGy, int* h_splatIdx, int* d_splatIdx,
                               float* h_pcloud, float* d_pcloud, float* d_cost,float* d_std, float* d_xrayGx, float* d_xrayGy,
                               float* d_M, float* h_camera, int N, float xrayGxstd, float xrayGystd, float &h_finalCost)
{

    int W=(int)h_camera[0];
    int H=(int)h_camera[1];

    float h_std[HEIGHT];
    float h_cost[HEIGHT];

    float mean;

    int idx=0;
    int T=WIDTH;
    int B=N/T;
    int pxcount = W*H;
    float fpxcount = (float)pxcount;
    cudaError_t status;


    SPLAT_Kernel<<<B,T>>>(d_pcloud, d_M, d_splatIdx);
    status = cudaGetLastError();
    if (status != cudaSuccess){printf("Problem SPLAT kernel!\n");return status;}

    status = cudaMemcpy(h_splatIdx, d_splatIdx, N*sizeof(int), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess){printf("Copying cost back to the host failed!\n");return status;}



    memset(h_drr, 0, pxcount*sizeof(float) );
    for(int k=0; k<N; k++)
    {
        idx = h_splatIdx[k];
        if(idx >=0 & idx<pxcount)
        {
            h_drr[idx]+=h_pcloud[3*N+k];
        }

    }

    status = cudaMemcpy(d_drr, h_drr, pxcount*sizeof(float), cudaMemcpyHostToDevice);
    if (status != cudaSuccess){printf("Copying splat drr to the device failed!\n");return status;}

    status = EntryComputeGradient(d_drr, d_drrGx, d_drrGy);
    if (status != cudaSuccess){return status;}

    //Here we have to do each image
    status = ComputeImageMean(mean, d_drrGx,d_std);
    if (status != cudaSuccess){return status;}

    float mn[1];mn[0] = mean;
    status = cudaMemcpyToSymbol(d_DRRGMean, mn, sizeof(float));
    if(status!=cudaSuccess){printf("Error copying constant mean to GPU!\n");return status;}


    NGC1_Kernel<<<HEIGHT,WIDTH>>>(d_drrGx, d_xrayGx, d_cost, d_std);
    status = cudaGetLastError();
    if (status != cudaSuccess){printf("Problem NGC cost function kernel!\n");return status;}

    status = cudaMemcpy(h_cost, d_cost, HEIGHT*sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess){printf("Copying cost to the host failed!\n");return status;}
    status = cudaMemcpy(h_std, d_std, HEIGHT*sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess){printf("Copying std vals to the host failed!\n");return status;}

    float drrstd = 0.0;
    float nccGx = 0.0;
    for( int k=0; k<H; k++)
    {
        drrstd += h_std[k];
        nccGx  += h_cost[k];
    }
    drrstd=sqrt(drrstd/fpxcount);
    nccGx=nccGx/fpxcount;
    nccGx=nccGx/(drrstd*xrayGxstd);
    nccGx = 1-nccGx;

    //Now for the second image
    status = ComputeImageMean(mean, d_drrGy,d_std);
    if (status != cudaSuccess){return status;}

    mn[0] = mean;
    status = cudaMemcpyToSymbol(d_DRRGMean, mn, sizeof(float));
    if(status!=cudaSuccess){printf("Error copying constant mean to GPU!\n");return status;}


    NGC1_Kernel<<<HEIGHT,WIDTH>>>(d_drrGy, d_xrayGy, d_cost, d_std);
    status = cudaGetLastError();
    if (status != cudaSuccess){printf("Problem NGC cost function kernel!\n");return status;}

    status = cudaMemcpy(h_cost, d_cost, HEIGHT*sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess){printf("Copying cost to the host failed!\n");return status;}
    status = cudaMemcpy(h_std, d_std, HEIGHT*sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess){printf("Copying std vals to the host failed!\n");return status;}


    float nccGy = 0.0;
    for( int k=0; k<H; k++)
    {
        drrstd += h_std[k];
        nccGy  += h_cost[k];
    }
    drrstd=sqrt(drrstd/fpxcount);
    nccGy=nccGy/fpxcount;
    nccGy=nccGy/(drrstd*xrayGystd);
    nccGy = 1-nccGy;

    h_finalCost = 0.5*nccGx + 0.5*nccGy;
    //printf("ncc %f \n",ncc);
    //printf("drrstd %f \n",drrstd);
    //printf("xrystd %f \n",xraystd);


    return status;


}

cudaError_t EntrySetConstantDataSPLAT(float* camera1, float* camera2, float pcsize,
                                       float xraymean1, float xraymean2, float drrmean)
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

    float xm1[1];xm1[0]=xraymean1;
    float xm2[1];xm2[0]=xraymean2;
    float dm[1];dm[0]=drrmean;

	status = cudaMemcpyToSymbol(d_camera1, camera1, sizeof(float)*12);
	if(status!=cudaSuccess){printf("Error copying camera parameters to GPU!\n");return status;}	
	status = cudaMemcpyToSymbol(d_camera2, camera2, sizeof(float)*12);
	if(status!=cudaSuccess){printf("Error copying camera parameters to GPU!\n");return status;}	
	status = cudaMemcpyToSymbol(d_invcamera1, inv1, sizeof(float)*12);
	if(status!=cudaSuccess){printf("Error copying camera parameters to GPU!\n");return status;}	
	status = cudaMemcpyToSymbol(d_invcamera2, inv2, sizeof(float)*12);
	if(status!=cudaSuccess){printf("Error copying camera parameters to GPU!\n");return status;}

    status = cudaMemcpyToSymbol(d_XRAY1Mean, xm1, sizeof(float));
    if(status!=cudaSuccess){printf("Error copying xray1 mean to GPU!\n");return status;}
    status = cudaMemcpyToSymbol(d_XRAY1Mean, xm2, sizeof(float));
    if(status!=cudaSuccess){printf("Error copying xray2 mean to GPU!\n");return status;}
    status = cudaMemcpyToSymbol(d_DRRMean,   dm, sizeof(float));
    if(status!=cudaSuccess){printf("Error copying drr mean to GPU!\n");return status;}

	status = cudaMemcpyToSymbol(d_pcParams, pc, sizeof(int));
	if(status!=cudaSuccess){printf("Error copying point cloud parameters to GPU!\n");return status;}

	return status;
}

#include "RayCast.h"


cudaError_t CudaEntryRayCast1(float* h_Cam2CT, int* h_BoundingBox, float* h_rayParams, float* d_DRR);
cudaError_t CudaEntryBindCT(cudaArray* d_CT, cudaChannelFormatDesc chDescCT);
cudaError_t CudaEntrySetConstantData1(float* camera, float xraymean);
cudaError_t CudaEntrySetConstantData2(float* camera, float xraymean);

cudaError_t CudaEntryNCC1(float* h_Cam2CT, int* h_BoundingBox, float* h_rayParams, float* h_std, float h_xraystd,
                          float* d_DRR, float* d_xray, float* d_vector1, float* d_vector2, float* d_mean, float* d_ncc, float* d_std,
                          float* h_cost);

cudaError_t CudaEntryNCC2(float* h_Cam2CT, int* h_BoundingBox, float* h_rayParams, float* h_std, float h_xraystd,
                          float* d_DRR, float* d_xray, float* d_vector1, float* d_vector2, float* d_mean, float* d_ncc, float* d_std,
                          float* h_cost);

cudaError_t CudaEntryNGC1(float* h_Cam2CT, int* h_BoundingBox, float* h_rayParams, float* h_std, float h_xrayGxstd, float h_xrayGystd,
                          float* d_DRR, float* d_DRRGx, float* d_DRRGy, float* d_xrayGx, float* d_xrayGy,
                          float* d_vector1, float* d_vector2, float* d_mean, float* d_ncc, float* d_std, float* h_cost);

cudaError_t CudaEntryFilterImage1(float* d_img, float* d_Gx, float* d_Gy);
cudaError_t CudaEntryFilterImage2(float* d_img, float* d_Gx, float* d_Gy);



RayCast::RayCast() {

    cpuStatus=0;
    //Initialize the GPU
    StartUpGPU();
    if(cudaStatus != cudaSuccess){printf("Problem Starting Up the GPU!\n");return;}

    //Initialize CT
    h_CT   = (float*)malloc(sizeof(float));

    //Initialize X-ray and DRRs
    h_xray1     = (float*)malloc(sizeof(float));
    h_xray2     = (float*)malloc(sizeof(float));
    h_xrayGx1     = (float*)malloc(sizeof(float));
    h_xrayGx2     = (float*)malloc(sizeof(float));
    h_xrayGy1     = (float*)malloc(sizeof(float));
    h_xrayGy2     = (float*)malloc(sizeof(float));
    h_DRR1      = (float*)malloc(sizeof(float));
    h_DRR2      = (float*)malloc(sizeof(float));
    h_DRRGx1    = (float*)malloc(sizeof(float));
    h_DRRGx2    = (float*)malloc(sizeof(float));
    h_DRRGy1    = (float*)malloc(sizeof(float));
    h_DRRGy2    = (float*)malloc(sizeof(float));

    cudaStatus = cudaMalloc((void **) &d_DRR1,   sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the DRR failed!\n");return;}
    cudaStatus = cudaMalloc((void **) &d_DRRGx1, sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the DRR x-gradient failed!\n");return;}
    cudaStatus = cudaMalloc((void **) &d_DRRGy1, sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the DRR y-gradient failed!\n");return;}
    cudaStatus = cudaMalloc((void **) &d_xray1,   sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray image failed!\n");return;}
    cudaStatus = cudaMalloc((void **) &d_xrayGx1, sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray x-gradient image failed!\n");return;}
    cudaStatus = cudaMalloc((void **) &d_xrayGy1, sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray y-gradient image failed!\n");return;}

    cudaStatus = cudaMalloc((void **) &d_DRR2,   sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the DRR failed!\n");return;}
    cudaStatus = cudaMalloc((void **) &d_DRRGx2, sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the DRR x-gradient failed!\n");return;}
    cudaStatus = cudaMalloc((void **) &d_DRRGy2, sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the DRR y-gradient failed!\n");return;}
    cudaStatus = cudaMalloc((void **) &d_xray2,   sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray image failed!\n");return;}
    cudaStatus = cudaMalloc((void **) &d_xrayGx2, sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray x-gradient image failed!\n");return;}
    cudaStatus = cudaMalloc((void **) &d_xrayGy2, sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray y-gradient image failed!\n");return;}

    cudaStatus = cudaMalloc((void **) &d_vector1, 512*sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating a column of data failed!\n");return;}
    cudaStatus = cudaMalloc((void **) &d_vector2, 512*sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating a column of data failed!\n");return;}
    cudaStatus = cudaMalloc((void **) &d_cost, sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating ncc failed!\n");return;}
    cudaStatus = cudaMalloc((void **) &d_mean, sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating mean failed!\n");return;}
    cudaStatus = cudaMalloc((void **) &d_std, sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating std failed!\n");return;}


    channelDescCT = cudaCreateChannelDesc<float>();
    volSize = make_cudaExtent(1, 1, 1);
    volSize.width  = 1;
    volSize.height = 1;
    volSize.depth  = 1;
    cudaStatus=cudaMalloc3DArray(&d_CT, &channelDescCT, volSize);
    if(cudaStatus!=cudaSuccess){printf("Allocating the CT failed!\n");return;}

}

RayCast::~RayCast()
{
    free(h_xray1);
    free(h_xray2);
    free(h_DRR1);
    free(h_DRRGx1);
    free(h_DRRGy1);
    free(h_DRR2);
    free(h_DRRGx2);
    free(h_DRRGy2);
    free(h_CT);

    cudaFree(d_xray1);
    cudaFree(d_xrayGx1);
    cudaFree(d_xrayGy1);
    cudaFree(d_xray2);
    cudaFree(d_xrayGx2);
    cudaFree(d_xrayGy2);
    cudaFree(d_DRR1);
    cudaFree(d_DRRGx1);
    cudaFree(d_DRRGy1);
    cudaFree(d_DRR2);
    cudaFree(d_DRRGx2);
    cudaFree(d_DRRGy2);
    cudaFreeArray(d_CT);

    cudaFree(d_vector1);
    cudaFree(d_vector2);
    cudaFree(d_mean);
    cudaFree(d_cost);
    cudaFree(d_std);

}

int RayCast::GetNumXray()
{
    return numXray;
}


float RayCast::ComputeNCC()
{

    UpdateTransformMatrix();
    ComputeBoundingBox1();

    cudaStatus =  CudaEntryNCC1(h_Cam2CT1, h_BoundingBox1, h_rayParams1, h_std, h_xraystd1,
                                d_DRR1, d_xray1, d_vector1, d_vector2, d_mean, d_cost, d_std,
                                h_cost);
    nccVal = h_cost[0];

    if(numXray==1)
    {
        return nccVal;
    }
    else if(numXray==2)
    {
        ComputeBoundingBox2();
        cudaStatus =  CudaEntryNCC2(h_Cam2CT2, h_BoundingBox2, h_rayParams2, h_std, h_xraystd2,
                                    d_DRR2, d_xray2, d_vector1, d_vector2, d_mean, d_cost, d_std,
                                    h_cost);
        nccVal = 0.5*nccVal + 0.5*h_cost[0];
        return nccVal;
    }
    return 0;

}

float RayCast::ComputeNGC()
{

    UpdateTransformMatrix();
    ComputeBoundingBox1();

    cudaStatus =  CudaEntryNGC1(h_Cam2CT1, h_BoundingBox1, h_rayParams1, h_std, h_xrayGxstd1, h_xrayGystd1,
                              d_DRR1, d_DRRGx1, d_DRRGy1, d_xrayGx1, d_xrayGy1,
                              d_vector1, d_vector2, d_mean, d_cost, d_std, h_cost);

    ngcVal = h_cost[0];



    if(numXray==1)
    {
        return ngcVal;
    }

    else if(numXray==2)
    {
        ComputeBoundingBox2();

        ///camera 2
        cudaStatus =  CudaEntryNGC1(h_Cam2CT2, h_BoundingBox2, h_rayParams2, h_std, h_xrayGxstd2, h_xrayGystd2,
                                  d_DRR2, d_DRRGx2, d_DRRGy2, d_xrayGx2, d_xrayGy2,
                                  d_vector1, d_vector2, d_mean, d_cost, d_std, h_cost);

        ngcVal = 0.5*ngcVal + 0.5*h_cost[0];

        return ngcVal;
    }
    return 0;

}


void RayCast::CreateRayCastGPU()
{		
    UpdateTransformMatrix();
    ComputeBoundingBox1();
    cudaStatus =  CudaEntryRayCast1(h_Cam2CT1, h_BoundingBox1, h_rayParams1, d_DRR1);
}

void RayCast::StartUpGPU() {

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess){
        	printf("InitGPU: Setting the Device to 0 failed!\n");
		return;
    	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess){
        	printf("InitGPU: Resetting the Device to 0 failed!\n");
		return;
    	}
	if (cudaStatus == cudaSuccess){
        	printf("InitGPU: GPU Started up OK!\n");
    	}

}

void RayCast::ReadImage(std::string filename, float* img, int W, int H)
{
	FILE *fp;
    fp = fopen(filename.c_str(), "r");
	if(fp==NULL){
        printf("Error reading: %s\n",filename.c_str());
		cpuStatus = -1;	
		return;
	}
	size_t r = fread(img, sizeof(float), W*H, fp);
	fclose(fp);
	return;
}

void RayCast::ReadVolume(std::string filename, float* vol, int sx, int sy, int sz)
{
    FILE *fp;
    fp = fopen(filename.c_str(), "r");
	if(fp==NULL){
        printf("Error reading: %s\n",filename.c_str());
		cpuStatus = -1;	
		return;
	}
	size_t r = fread(vol, sizeof(float), sx*sy*sz, fp);
	fclose(fp);
	return;
}

void RayCast::WriteVolume(char* filename, float* vol, int sx, int sy, int sz)
{
	FILE *fp;
	fp = fopen(filename, "w");
	if(fp==NULL){
        printf("Error writing: %s\n",filename);
		cpuStatus = -2;	
		return;
	}
	size_t r = fwrite(vol, sizeof(float), sx*sy*sz, fp);
	fclose(fp);
	return;
}

void RayCast::WriteImage(char* filename, float* img, int W, int H)
{
	FILE *fp;
	fp = fopen(filename, "w");
	if(fp==NULL){
        printf("Error writing: %s\n",filename);
		cpuStatus = -2;	
		return;
	}
	size_t r = fwrite(img, sizeof(float), W*H, fp);
	fclose(fp);
	return;
}

void RayCast::TransferXrayToHost1()
{
    int W=(int)h_camera1[0];
    int H=(int)h_camera1[1];
    cudaError_t cudaStatus;
    cudaStatus = cudaMemcpy(h_xray1, d_xray1, H*W*sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        printf("Transfering the xray from the GPU failed!\n");
    }
}

void RayCast::TransferXrayGradientToHost1()
{
    int W=(int)h_camera1[0];
    int H=(int)h_camera1[1];
    cudaError_t status;
    status = cudaMemcpy(h_xrayGx1, d_xrayGx1, H*W*sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        printf("Transfering the xray x-gradient from the GPU failed!\n");
        return;
    }
    status = cudaMemcpy(h_xrayGy1, d_xrayGy1, H*W*sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        printf("Transfering the xray y-gradient from the GPU failed!\n");
        return;
    }

}

void RayCast::TransferXrayToHost2()
{
    int W=(int)h_camera2[0];
    int H=(int)h_camera2[1];
    cudaError_t cudaStatus;
    cudaStatus = cudaMemcpy(h_xray2, d_xray2, H*W*sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        printf("Transfering the xray from the GPU failed!\n");
    }
}

void RayCast::TransferXrayGradientToHost2()
{
    int W=(int)h_camera2[0];
    int H=(int)h_camera2[1];
    cudaError_t status;
    status = cudaMemcpy(h_xrayGx2, d_xrayGx2, H*W*sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        printf("Transfering the xray x-gradient from the GPU failed!\n");
        return;
    }
    status = cudaMemcpy(h_xrayGy2, d_xrayGy2, H*W*sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        printf("Transfering the xray y-gradient from the GPU failed!\n");
        return;
    }

}

void RayCast::TransferDRRToHost1()
{
    int W=(int)h_camera1[0];
    int H=(int)h_camera1[1];
	cudaError_t cudaStatus;
    cudaStatus = cudaMemcpy(h_DRR1, d_DRR1, H*W*sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
    {
        printf("Transfering the DRR from the GPU failed!\n");
    }
}

void RayCast::TransferDRRGradientToHost1()
{
    int W=(int)h_camera1[0];
    int H=(int)h_camera1[1];
	cudaError_t status;
    status = cudaMemcpy(h_DRRGx1, d_DRRGx1, H*W*sizeof(float), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) 
	{
        printf("Transfering the DRR1 x-gradient from the GPU failed!\n");
		return;
    }
    status = cudaMemcpy(h_DRRGy1, d_DRRGy1, H*W*sizeof(float), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) 
	{
        printf("Transfering the DRR1 y-gradient from the GPU failed!\n");
		return;
    }

}

void RayCast::TransferDRRToHost2()
{
    int W=(int)h_camera2[0];
    int H=(int)h_camera2[1];
    cudaError_t cudaStatus;
    cudaStatus = cudaMemcpy(h_DRR2, d_DRR2, H*W*sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        printf("Transfering the DRR from the GPU failed!\n");
    }
}

void RayCast::TransferDRRGradientToHost2()
{
    int W=(int)h_camera2[0];
    int H=(int)h_camera2[1];
    cudaError_t status;
    status = cudaMemcpy(h_DRRGx2, d_DRRGx2, H*W*sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        printf("Transfering the DRR1 x-gradient from the GPU failed!\n");
        return;
    }
    status = cudaMemcpy(h_DRRGy2, d_DRRGy2, H*W*sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        printf("Transfering the DRR1 y-gradient from the GPU failed!\n");
        return;
    }

}



void RayCast::SetCTParams(float xdim, float ydim, float zdim, float dx, float dy, float dz, float cx, float cy, float cz)
{
	//We have to invert the pixel spacing for the CT so that multiplications can be performed, rather than divides
	//Within the cuda kernel
    h_ctParams[0]=xdim;
    h_ctParams[1]=ydim;
    h_ctParams[2]=zdim;
    h_ctParams[3]=dx;
    h_ctParams[4]=dy;
    h_ctParams[5]=dz;
    h_ctParams[6]=cx;
    h_ctParams[7]=cy;
    h_ctParams[8]=cz;

    //Free previously allocates stuff
    free(h_CT);
    cudaFreeArray(d_CT);

    h_CT = (float*)malloc(xdim*ydim*zdim*sizeof(float));
    volSize = make_cudaExtent((int)xdim, (int)ydim, (int)zdim);
    volSize.width  = (int)xdim;
    volSize.height = (int)ydim;
    volSize.depth  = (int)zdim;
    cudaStatus=cudaMalloc3DArray(&d_CT, &channelDescCT, volSize);
    if(cudaStatus!=cudaSuccess){printf("Allocating the CT failed!\n");return;}

}

void RayCast::ReadXray1(std::string filename, float W, float H)
{

    AllocateXray1();
    ReadImage(filename, h_xray1, (int)W, (int)H);
    h_xraymean1 = ComputeImageMean(h_xray1,(int)W, (int)H);
    h_xraystd1  = ComputeImageStd(h_xray1, h_xraymean1, (int)W, (int)H);
    TransferXrayGPU1();
    cudaStatus =  CudaEntrySetConstantData1(h_camera1, h_xraymean1);
    if(cudaStatus!=cudaSuccess)
    {
        return;
    }

    CudaEntryFilterImage1(d_xray1, d_xrayGx1, d_xrayGy1);
    TransferXrayGradientToHost1();
    h_xrayGxmean1 = ComputeImageMean(h_xrayGx1,(int)W, (int)H);
    h_xrayGxstd1  = ComputeImageStd(h_xrayGx1, h_xrayGxmean1, (int)W, (int)H);
    h_xrayGymean1 = ComputeImageMean(h_xrayGy1,(int)W, (int)H);
    h_xrayGystd1  = ComputeImageStd(h_xrayGy1, h_xrayGymean1, (int)W, (int)H);

    printf("Xray 1 statistics:\n");
    printf("   Mean %f StdDev %f:\n",h_xraymean1,h_xraystd1);
    printf("Gx Mean %f StdDev %f:\n",h_xrayGxmean1,h_xrayGxstd1);
    printf("Gy Mean %f StdDev %f:\n",h_xrayGymean1,h_xrayGystd1);

}

void RayCast::ReadXray2(std::string filename, float W, float H)
{
    AllocateXray2();
    ReadImage(filename, h_xray2, (int)W, (int)H);
    h_xraymean2 = ComputeImageMean(h_xray2,(int)W, (int)H);
    h_xraystd2  = ComputeImageStd(h_xray2, h_xraymean2, (int)W, (int)H);
    TransferXrayGPU2();
    cudaStatus =  CudaEntrySetConstantData2(h_camera2, h_xraymean2);
    if(cudaStatus!=cudaSuccess)
    {
        return;
    }
    CudaEntryFilterImage2(d_xray2, d_xrayGx2, d_xrayGy2);
    TransferXrayGradientToHost2();
    h_xrayGxmean2 = ComputeImageMean(h_xrayGx2,(int)W, (int)H);
    h_xrayGxstd2  = ComputeImageStd(h_xrayGx2, h_xrayGxmean2, (int)W, (int)H);
    h_xrayGymean2 = ComputeImageMean(h_xrayGy2,(int)W, (int)H);
    h_xrayGystd2  = ComputeImageStd(h_xrayGy2, h_xrayGymean2, (int)W, (int)H);

    printf("Xray 2 statistics:\n");
    printf("   Mean %f StdDev %f:\n",h_xraymean1,h_xraystd1);
    printf("Gx Mean %f StdDev %f:\n",h_xrayGxmean1,h_xrayGxstd1);
    printf("Gy Mean %f StdDev %f:\n",h_xrayGymean1,h_xrayGystd1);
}

void RayCast::ReadCT(std::string filename)
{

    CreateCTBoundingBox();
    AllocateCT();
    ReadVolume(filename, h_CT, (int)h_ctParams[0], (int)h_ctParams[1], (int)h_ctParams[2]);
	TransferCTGPU();
}

void RayCast::TransferXrayGPU1()
{
    int W=(int)h_camera1[0];
    int H=(int)h_camera1[1];
    cudaStatus = cudaMemcpy(d_xray1, h_xray1, W*H*sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess){printf("Transfering the X-ray image to the GPU failed!\n");}
}

void RayCast::TransferXrayGPU2()
{
    int W=(int)h_camera2[0];
    int H=(int)h_camera2[1];
    cudaStatus = cudaMemcpy(d_xray2, h_xray2, W*H*sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess){printf("Transfering the X-ray image to the GPU failed!\n");}
}

void RayCast::TransferCTGPU()
{

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void *)h_CT, volSize.width*sizeof(float),volSize.width, volSize.height);
    copyParams.dstArray = d_CT;
    copyParams.extent = volSize;
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaStatus = cudaMemcpy3D(&copyParams);

    if(cudaStatus!=cudaSuccess)
    {
        printf("Error copying CT to GPU!\n");
        return;
    }
    cudaStatus = CudaEntryBindCT(d_CT, channelDescCT);

}


void RayCast::TransferConstantsToGPU()
{

    cudaStatus =  CudaEntrySetConstantData1(h_camera1, h_xraymean1);
    if(cudaStatus!=cudaSuccess)
    {
        return;
    }
    cudaStatus =  CudaEntrySetConstantData2(h_camera2, h_xraymean2);
    if(cudaStatus!=cudaSuccess)
    {
        return;
    }

}

void RayCast::WriteDRR(char* filename, int num){

    if(num==1)
    {
        WriteImage(filename,h_DRR1, (int)h_camera1[0], (int)h_camera1[1]);
    }
    else
    {
        WriteImage(filename,h_DRR2, (int)h_camera2[0], (int)h_camera2[1]);
    }
}

void RayCast::WriteDRRGx(char* filename, int num){

    if(num==1)
    {
        WriteImage(filename,h_DRRGx1, (int)h_camera1[0], (int)h_camera1[1]);
    }
    else
    {
        WriteImage(filename,h_DRRGx2, (int)h_camera2[0], (int)h_camera2[1]);
    }
}

void RayCast::WriteDRRGy(char* filename, int num){

    if( num==1)
    {
        WriteImage(filename,h_DRRGy1, (int)h_camera1[0], (int)h_camera1[1]);
    }
    else
    {
        WriteImage(filename,h_DRRGy2, (int)h_camera2[0], (int)h_camera2[1]);
    }
}

void RayCast::WriteXrayGx(char* filename, int num){

    if(num==1)
    {
        WriteImage(filename,h_xrayGx1, (int)h_camera1[0], (int)h_camera1[1]);
    }
    else
    {
        WriteImage(filename,h_xrayGy2, (int)h_camera2[0], (int)h_camera2[1]);
    }
}

void RayCast::WriteXrayGy(char* filename, int num){

    if( num==1)
    {
        WriteImage(filename,h_xrayGy1, (int)h_camera1[0], (int)h_camera1[1]);
    }
    else
    {
        WriteImage(filename,h_xrayGy2, (int)h_camera2[0], (int)h_camera2[1]);
    }
}

bool RayCast::isOK(){
	if(cudaStatus == cudaSuccess & cpuStatus==0)
	{return true;}
	else if(cudaStatus != cudaSuccess)
	{
		printf("CUDA Error code: %d\n",cudaStatus);
		return false;
	}
	else
	{
        printf("CPU Error code: %d\n",cpuStatus);
		return false;
	}
}

void RayCast::SetInitialTransform(float tx, float ty, float tz, float rx, float ry, float rz)
{

    h_initial[0] = tx;
    h_initial[1] = ty;
    h_initial[2] = tz;
    h_initial[3] = rx;
    h_initial[4] = ry;
    h_initial[5] = rz;

}

void RayCast::SetSecondaryTransform(float tx, float ty, float tz, float rx, float ry, float rz)
{
    h_secondary[0] = tx;
    h_secondary[1] = ty;
    h_secondary[2] = tz;
    h_secondary[3] = rx;
    h_secondary[4] = ry;
    h_secondary[5] = rz;
}

void RayCast::SetSpatialTranslation(float (&M)[16],float* tform)
{
    M[3]=tform[0];
    M[7]=tform[1];
    M[11]=tform[2];
}

void RayCast::SetSpatialTranslationInverse(float (&M)[16],float* tform)
{
    M[3]=-tform[0];
    M[7]=-tform[1];
    M[11]=-tform[2];
}

void RayCast::SetSpatialRotation(float (&M)[16],float* tform)
{
    float cx = cos(tform[3]);float sx = sin(tform[3]);
    float cy = cos(-tform[4]);float sy = sin(-tform[4]);
    float cz = cos(tform[5]);float sz = sin(tform[5]);

    M[0] = cy*cz+sx*sy*sz; M[1] = -cx*sz; M[2] = cy*sx*sz-cz*sy;
    M[4] = cy*sz-cz*sx*sy; M[5] =  cx*cz; M[6] = -sy*sz-cy*cz*sx;
    M[8] = cx*sy;          M[9] =  sx;    M[10] = cx*cy;

}

void RayCast::SetSpatialRotationInverse(float (&M)[16],float* tform)
{
    float cx = cos(tform[3]);float sx = sin(tform[3]);
    float cy = cos(-tform[4]);float sy = sin(-tform[4]);
    float cz = cos(tform[5]);float sz = sin(tform[5]);

    M[0] = cy*cz+sx*sy*sz; M[1] = cy*sz-cz*sx*sy;  M[2] = cx*sy;
    M[4] = -cx*sz;         M[5] = cx*cz;           M[6] = sx;
    M[8] = cy*sx*sz-cz*sy; M[9] = -sy*sz-cy*cz*sx; M[10] = cx*cy;


}

void RayCast::UpdateTransformMatrix()
{

    float sT[16]; SetEye(sT);
    float sR[16]; SetEye(sR);
    float sTinv[16]; SetEye(sTinv);
    float sRinv[16]; SetEye(sRinv);

    SetSpatialTranslation(sT,h_secondary);
    SetSpatialRotation(sR,h_secondary);
    SetSpatialTranslationInverse(sTinv,h_secondary);
    SetSpatialRotationInverse(sRinv,h_secondary);


    MM44(TMP1,   sR,        h_iR);
    MM44(TMP2,    h_iT,        TMP1);
    MM44(h_P2W,  sT, TMP2);
    MM44(TMP1, h_P2W, h_CT2P);
    MM44(h_CT2Cam1, h_W2Cam1, TMP1);


    MM44(TMP1,   h_iTinv,        sTinv);
    MM44(TMP2,    sRinv,        TMP1);
    MM44(h_W2P,  h_iRinv, TMP2);
    MM44(TMP1, h_W2P, h_Cam2W1);
    MM44(h_Cam2CT1, h_P2CT, TMP1);



    if(numXray==2)
    {
        MM44(TMP1,   sR,        h_iR);
        MM44(TMP2,    h_iT,        TMP1);
        MM44(h_P2W,  sT, TMP2);
        MM44(TMP1, h_P2W, h_CT2P);
        MM44(h_CT2Cam2, h_W2Cam2, TMP1);

        MM44(TMP1,   h_iTinv,        sTinv);
        MM44(TMP2,    sRinv,        TMP1);
        MM44(h_W2P,  h_iRinv, TMP2);
        MM44(TMP1, h_W2P, h_Cam2W2);
        MM44(h_Cam2CT2, h_P2CT, TMP1);
    }

}

void RayCast::MM44(float (&MO)[16], float (&ML)[16], float (&MR)[16])
{
    MO[0]  = ML[0]*MR[0]  + ML[1]*MR[4]  + ML[2]*MR[8]   + ML[3]*MR[12];
    MO[1]  = ML[0]*MR[1]  + ML[1]*MR[5]  + ML[2]*MR[9]   + ML[3]*MR[13];
    MO[2]  = ML[0]*MR[2]  + ML[1]*MR[6]  + ML[2]*MR[10]  + ML[3]*MR[14];
    MO[3]  = ML[0]*MR[3]  + ML[1]*MR[7]  + ML[2]*MR[11]  + ML[3]*MR[15];

    MO[4]  = ML[4]*MR[0]  + ML[5]*MR[4]  + ML[6]*MR[8]   + ML[7]*MR[12];
    MO[5]  = ML[4]*MR[1]  + ML[5]*MR[5]  + ML[6]*MR[9]   + ML[7]*MR[13];
    MO[6]  = ML[4]*MR[2]  + ML[5]*MR[6]  + ML[6]*MR[10]  + ML[7]*MR[14];
    MO[7]  = ML[4]*MR[3]  + ML[5]*MR[7]  + ML[6]*MR[11]  + ML[7]*MR[15];

    MO[8]  = ML[8]*MR[0]  + ML[9]*MR[4]  + ML[10]*MR[8]  + ML[11]*MR[12];
    MO[9]  = ML[8]*MR[1]  + ML[9]*MR[5]  + ML[10]*MR[9]  + ML[11]*MR[13];
    MO[10] = ML[8]*MR[2]  + ML[9]*MR[6]  + ML[10]*MR[10] + ML[11]*MR[14];
    MO[11] = ML[8]*MR[3]  + ML[9]*MR[7]  + ML[10]*MR[11] + ML[11]*MR[15];

    MO[12] = ML[12]*MR[0] + ML[13]*MR[4] + ML[14]*MR[8]  + ML[15]*MR[12];
    MO[13] = ML[12]*MR[1] + ML[13]*MR[5] + ML[14]*MR[9]  + ML[15]*MR[13];
    MO[14] = ML[12]*MR[2] + ML[13]*MR[6] + ML[14]*MR[10] + ML[15]*MR[14];
    MO[15] = ML[12]*MR[3] + ML[13]*MR[7] + ML[14]*MR[11] + ML[15]*MR[15];
}

void RayCast::MM48(float (&MO)[32], float (&ML)[16], float (&MR)[32])
{

    MO[0]   = ML[0]*MR[0]  + ML[1]*MR[8]  + ML[2]*MR[16] + ML[3]*MR[24];
    MO[1]   = ML[0]*MR[1]  + ML[1]*MR[9]  + ML[2]*MR[17] + ML[3]*MR[25];
    MO[2]   = ML[0]*MR[2]  + ML[1]*MR[10] + ML[2]*MR[18] + ML[3]*MR[26];
    MO[3]   = ML[0]*MR[3]  + ML[1]*MR[11] + ML[2]*MR[19] + ML[3]*MR[27];
    MO[4]   = ML[0]*MR[4]  + ML[1]*MR[12] + ML[2]*MR[20] + ML[3]*MR[28];
    MO[5]   = ML[0]*MR[5]  + ML[1]*MR[13] + ML[2]*MR[21] + ML[3]*MR[29];
    MO[6]   = ML[0]*MR[6]  + ML[1]*MR[14] + ML[2]*MR[22] + ML[3]*MR[30];
    MO[7]   = ML[0]*MR[7]  + ML[1]*MR[15] + ML[2]*MR[23] + ML[3]*MR[31];

    MO[8]   = ML[4]*MR[0]  + ML[5]*MR[8]  + ML[6]*MR[16] + ML[7]*MR[24];
    MO[9]   = ML[4]*MR[1]  + ML[5]*MR[9]  + ML[6]*MR[17] + ML[7]*MR[25];
    MO[10]  = ML[4]*MR[2]  + ML[5]*MR[10] + ML[6]*MR[18] + ML[7]*MR[26];
    MO[11]  = ML[4]*MR[3]  + ML[5]*MR[11] + ML[6]*MR[19] + ML[7]*MR[27];
    MO[12]  = ML[4]*MR[4]  + ML[5]*MR[12] + ML[6]*MR[20] + ML[7]*MR[28];
    MO[13]  = ML[4]*MR[5]  + ML[5]*MR[13] + ML[6]*MR[21] + ML[7]*MR[29];
    MO[14]  = ML[4]*MR[6]  + ML[5]*MR[14] + ML[6]*MR[22] + ML[7]*MR[30];
    MO[15]  = ML[4]*MR[7]  + ML[5]*MR[15] + ML[6]*MR[23] + ML[7]*MR[31];

    MO[16]  = ML[8]*MR[0]  + ML[9]*MR[8]  + ML[10]*MR[16] + ML[11]*MR[24];
    MO[17]  = ML[8]*MR[1]  + ML[9]*MR[9]  + ML[10]*MR[17] + ML[11]*MR[25];
    MO[18]  = ML[8]*MR[2]  + ML[9]*MR[10] + ML[10]*MR[18] + ML[11]*MR[26];
    MO[19]  = ML[8]*MR[3]  + ML[9]*MR[11] + ML[10]*MR[19] + ML[11]*MR[27];
    MO[20]  = ML[8]*MR[4]  + ML[9]*MR[12] + ML[10]*MR[20] + ML[11]*MR[28];
    MO[21]  = ML[8]*MR[5]  + ML[9]*MR[13] + ML[10]*MR[21] + ML[11]*MR[29];
    MO[22]  = ML[8]*MR[6]  + ML[9]*MR[14] + ML[10]*MR[22] + ML[11]*MR[30];
    MO[23]  = ML[8]*MR[7]  + ML[9]*MR[15] + ML[10]*MR[23] + ML[11]*MR[31];

    MO[24]  = ML[12]*MR[0] + ML[13]*MR[8]  + ML[14]*MR[16] + ML[15]*MR[24];
    MO[25]  = ML[12]*MR[1] + ML[13]*MR[9]  + ML[14]*MR[17] + ML[15]*MR[25];
    MO[26]  = ML[12]*MR[2] + ML[13]*MR[10] + ML[14]*MR[18] + ML[15]*MR[26];
    MO[27]  = ML[12]*MR[3] + ML[13]*MR[11] + ML[14]*MR[19] + ML[15]*MR[27];
    MO[28]  = ML[12]*MR[4] + ML[13]*MR[12] + ML[14]*MR[20] + ML[15]*MR[28];
    MO[29]  = ML[12]*MR[5] + ML[13]*MR[13] + ML[14]*MR[21] + ML[15]*MR[29];
    MO[30]  = ML[12]*MR[6] + ML[13]*MR[14] + ML[14]*MR[22] + ML[15]*MR[30];
    MO[31]  = ML[12]*MR[7] + ML[13]*MR[15] + ML[14]*MR[23] + ML[15]*MR[31];

}

void RayCast::InitForwardMatrices()
{
    float D[16];
    float isoF[16];
    float isoI[16];
    float cR[16];
    float cT[16];
    float O[16];
    float S[16];

    SetEye(D);
    SetEye(isoF);
    SetEye(isoI);
    SetEye(O);
    SetEye(S);
    SetEye(h_iT);
    SetEye(h_iR);

    SetSpatialTranslation(h_iT,h_initial);
    SetSpatialRotation(h_iR,h_initial);

    O[3]     = -h_ctParams[6];
    O[7]     = -h_ctParams[7];
    O[11]    = -h_ctParams[8];
    S[0]     =  h_ctParams[3];
    S[5]     =  h_ctParams[4];
    S[10]    =  h_ctParams[5];

    MM44(h_CT2P,O,S);
    ///////////Camera 1///////////////
    for(int k=0; k<6;k++)
    {
        D[k] = h_camera1[6+k];
    }

    SetEye(cT);
    SetEye(cR);
    SetSpatialTranslationInverse(cT,D);
    SetSpatialRotationInverse(cR,D);

    isoF[11] = h_camera1[5];
    isoI[11] = -h_camera1[5];

    MM44(TMP1,cT,isoI);
    MM44(TMP2,cR,TMP1);
    MM44(h_W2Cam1,isoF,TMP2);

    ///////////Camera 2///////////////
    for(int k=0; k<6;k++)
    {
        D[k] = h_camera2[6+k];
    }

    SetEye(cT);
    SetEye(cR);
    SetSpatialTranslationInverse(cT,D);
    SetSpatialRotationInverse(cR,D);

    isoF[11] = h_camera2[5];
    isoI[11] = -h_camera2[5];

    MM44(TMP1,cT,isoI);
    MM44(TMP2,cR,TMP1);
    MM44(h_W2Cam2,isoF,TMP2);

}

void RayCast::InitInverseMatrices()
{
    float D[16];
    float isoF[16];
    float isoI[16];
    float cR[16];
    float cT[16];
    float O[16];
    float S[16];

    SetEye(D);
    SetEye(isoF);
    SetEye(isoI);
    SetEye(O);
    SetEye(S);
    SetEye(h_iTinv);
    SetEye(h_iRinv);

    SetSpatialTranslationInverse(h_iTinv,h_initial);
    SetSpatialRotationInverse(h_iRinv,h_initial);

    O[3]     = h_ctParams[6];
    O[7]     = h_ctParams[7];
    O[11]    = h_ctParams[8];
    S[0]     = 1/h_ctParams[3];
    S[5]     = 1/h_ctParams[4];
    S[10]    = 1/h_ctParams[5];

    MM44(h_P2CT,S,O);
    ///////////Camera 1///////////////
    for(int k=0; k<6;k++)
    {
        D[k] = h_camera1[6+k];
    }

    SetEye(cT);
    SetEye(cR);
    SetSpatialTranslation(cT,D);
    SetSpatialRotation(cR,D);

    isoF[11] = h_camera1[5];
    isoI[11] = -h_camera1[5];

    MM44(TMP1,cR,isoI);
    MM44(TMP2,cT,TMP1);
    MM44(h_Cam2W1,isoF,TMP2);

    ///////////Camera 2///////////////
    for(int k=0; k<6;k++)
    {
        D[k] = h_camera2[6+k];
    }

    SetEye(cT);
    SetEye(cR);
    SetSpatialTranslation(cT,D);
    SetSpatialRotation(cR,D);

    isoF[11] = h_camera2[5];
    isoI[11] = -h_camera2[5];

    MM44(TMP1,cT,isoI);
    MM44(TMP2,cR,TMP1);
    MM44(h_Cam2W2,isoF,TMP2);
}

void RayCast::SetEye(float (&M)[16])
{
    for(int k=0; k<16; k++)
    {
        if(k==0 | k==5 | k==10 | k==15)
        {
            M[k]=1.0;
        }
        else
        {
            M[k]=0.0;
        }

    }
}

void RayCast::ReadParameterFile(std::string filename)
{
    std::ifstream infile(filename.c_str());
    if(!infile){printf("Couldn't open file: %s\n",filename.c_str()); cpuStatus=-1;return;}

    std::stringstream ss;
    std::string line;
    std::string param;
    std::string str;
    int count=1;

    while(infile)
    {

        std::getline(infile, line);
        ss.str("");
        ss.clear();
        ss << line;
        ss >> param;

        if( line == ""){param="";}

        if(param == "numxrays")
        {
            ss >> numXray;
            printf("Number of xrays: %d\n",numXray);
        }

        else if(param == "xray1")
        {
            ss >> str;
            printf("X-ray filename 1: %s\n",str.c_str());
            printf("Camera 1 params: ");
            for(int k=0;k<12;k++)
            {
                ss >> h_camera1[k];
                printf("%f | ",h_camera1[k]);
            }
            printf("\n");
            ReadXray1(str,h_camera1[0], h_camera1[1]);

        }

        else if(param == "xray2")
        {
            ss >> str;
            printf("X-ray filename 2: %s\n",str.c_str());
            printf("Camera 2 params: ");

            for(int k=0;k<12;k++)
            {
                ss >> h_camera2[k];
                printf("%f| ",h_camera2[k]);

            }
            printf("\n");
            ReadXray2(str,h_camera2[0],h_camera2[1]);

        }

        else if(param == "ct")
        {
            ss >> str;
            for( int k=0; k<9; k++)
            {
                ss >> h_ctParams[k];
            }
            ReadCT(str);

        }

        else if(param == "initialtransform")
        {
            printf("Initial transform: ");
            for(int k=0;k<6;k++)
            {
                ss >> h_initial[k];
                h_secondary[k]=0.0;
                printf("%f | ",h_initial[k]);
            }
            printf("\n");
        }

        else if(param == "raydelta")
        {

            ss >> h_rayParams1[3];
            h_rayParams2[3] = h_rayParams1[3];
            printf("Ray delta %f\n",h_rayParams1[3] );

        }

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess){printf("Problem reading parameter file: \n%s!\n",line.c_str());return;}
        count++;
    }

    infile.close();
    InitForwardMatrices();
    InitInverseMatrices();
}

void RayCast::PrintNGC()
{

    printf("NGC: %f \n",ngcVal);

}

void RayCast::PrintNCC()
{

    printf("NCC: %f \n",nccVal);

}

void RayCast::PrintM44(float (&M)[16])
{
    int cnt=0;
    printf("_____________________________\n");
    for(int i=0; i<4; i++)
    {
        for(int j=0; j<4; j++)
        {
            printf("%f\t",M[cnt]);
            cnt++;
        }
        printf("\n");
    }
    printf("_____________________________\n");
}

void RayCast::PrintM48(float (&M)[32])
{
    int cnt=0;
    printf("_____________________________\n");
    for(int i=0; i<4; i++)
    {
        for(int j=0; j<8; j++)
        {
            printf("%f\t",M[cnt]);
            cnt++;
        }
        printf("\n");
    }
    printf("_____________________________\n");
}

void RayCast::ComputeBoundingBox1()
{

    float BoxTransformed[32];

    int W = h_camera1[0];
    int H = h_camera1[1];

    MM48(BoxTransformed,h_CT2Cam1,h_BoxCT);

    float x=0.0;
    float y=0.0;
    float z=0.0;
    float SID = h_camera1[4];
    int xmin = W;
    int xmax = 0;
    int ymin = H;
    int ymax = 0;
    int zmin = SID;
    int zmax = 0;
    float px = h_camera1[2];
    float py = h_camera1[3];

    //PrintM48( h_BoxCT);
    //PrintM44( h_P2W);
    //PrintM48( BoxTransformed);
    //Get the bounding box in screen coords
    for( int k=0; k<8; k++)
    {

        x = BoxTransformed[k];
        y = BoxTransformed[k+8];
        z = BoxTransformed[k+16];


        x = x*(SID/(SID-z));
        y = y*(SID/(SID-z));
        x = x/px;
        y = y/py;

        x = x + (0.5*(float)W);
        y = y + (0.5*(float)H);

        if(x < (float)xmin)
        {
            xmin = (int)floor(x);
        }

        if(x > (float)xmax)
        {
            xmax = (int)ceil(x);
        }

        if(y < (float)ymin)
        {
            ymin = (int)floor(y);
        }

        if(y > (float)ymax)
        {
            ymax = (int)ceil(y);
        }

        if(z < zmin)
        {
            zmin = z;
        }
        if(z > zmax)
        {
            zmax = z;
        }
    }

    //printf("Bounding Box xmin %d xmax %d ymin %d ymax %d\n", xmin, xmax, ymin, ymax);
    //if any of the bounding box coordinates are outside of the image, clamp them to the border

    //printf("Bounding Box Left %d Right %d Top %d Bottom %d\n", xmin,xmax,ymin,ymax);

    if(xmin < 0)
    {
        xmin = 0;
    }
    if(xmax > W)
    {
        xmax = W;
    }
    if(ymin < 0)
    {
        ymin = 0;
    }
    if(ymax > H)
    {
        ymax = H;
    }

    int xdiff = xmax-xmin+1;
    int ydiff = ymax-ymin+1;

    h_BoundingBox1[0] = xmin;
    h_BoundingBox1[1] = xdiff;
    h_BoundingBox1[2] = ymin;
    h_BoundingBox1[3] = ydiff;
    h_rayParams1[0]   = zmin;
    h_rayParams1[1]   = zmax;
    h_rayParams1[2]   = ceil((zmax-zmin)/h_rayParams1[3]);

    //printf("Bounding Box Left %d W %d Top %d H %d\n", h_BoundingBox1[0],h_BoundingBox1[1],h_BoundingBox1[2],h_BoundingBox1[3]);

}

void RayCast::ComputeBoundingBox2()
{

    float BoxTransformed[32];

    int W = h_camera2[0];
    int H = h_camera2[1];

    MM48(BoxTransformed,h_CT2Cam2,h_BoxCT);

    float x=0.0;
    float y=0.0;
    float z=0.0;
    float SID = h_camera2[4];
    int xmin = W-17;
    int xmax = 0;
    int ymin = H-17;
    int ymax = 0;
    int zmin = SID;
    int zmax = 0;
    float px = h_camera2[2];
    float py = h_camera2[3];

    //PrintM48( h_BoxCT);
    //PrintM44( h_CT2Cam1);
    //PrintM48( BoxTransformed);
    //Get the bounding box in screen coords
    for( int k=0; k<8; k++)
    {

        x = BoxTransformed[k];
        y = BoxTransformed[k+8];
        z = BoxTransformed[k+16];

        x = x*(SID/(SID-z));
        y = y*(SID/(SID-z));
        x = x/px;
        y = y/py;

        x = x + (0.5*(float)W);
        y = y + (0.5*(float)H);

        if(x < (float)xmin)
        {
            xmin = (int)floor(x);
        }
        if(x > (float)xmax)
        {
            xmax = (int)ceil(x);
        }

        if(y < (float)ymin)
        {
            ymin = (int)floor(y);
        }
        if(y > (float)ymax)
        {
            ymax = (int)ceil(y);
        }

        if(z < zmin)
        {
            zmin = z;
        }
        else if(z > zmax)
        {
            zmax = z;
        }
    }

    //printf("Bounding Box xmin %d xmax %d ymin %d ymax %d\n", xmin, xmax, ymin, ymax);
    //if any of the bounding box coordinates are outside of the image, clamp them to the border


    if(xmin < 0)
    {
        xmin = 0;
    }
    if(xmax > W)
    {
        xmax = W;
    }
    if(ymin < 0)
    {
        ymin = 0;
    }
    if(ymax > H)
    {
        ymax = H;
    }

    int xdiff = xmax-xmin+1;
    int ydiff = ymax-ymin+1;

    h_BoundingBox2[0] = xmin;
    h_BoundingBox2[1] = xdiff;
    h_BoundingBox2[2] = ymin;
    h_BoundingBox2[3] = ydiff;
    h_rayParams2[0]   = zmin;
    h_rayParams2[1]   = zmax;
    h_rayParams2[2]   = ceil((zmax-zmin)/h_rayParams2[3]);

    //printf("Bounding Box Left %d W %d Top %d H %d\n", h_BoundingBox2[0],h_BoundingBox2[1],h_BoundingBox2[2],h_BoundingBox2[3]);

}

float RayCast::ComputeImageMean(float* img, int W, int H)
{
    int k=0;
    float sum = 0;
    float pxcount = (float)(W*H);
    for( int i=0; i < W; i++)
    {
        for( int j=0; j < H; j++)
        {
            sum+=img[k];
            k++;
        }
    }
    sum=sum/pxcount;
    return sum;

}

float RayCast::ComputeImageStd(float* img, float mean, int W, int H)
{
    int k=0;
    float sum = 0.0;
    float val = 0.0;
    float pxcount = (float)(W*H);
    for( int i=0; i < W; i++)
    {
        for( int j=0; j < H; j++)
        {
            val = img[k]-mean;
            sum+=(val*val);
            k++;
        }
    }
    sum=sqrt(sum/pxcount);

    return sum;
}

void RayCast::AllocateXray1()
{

    int numpx = (int)h_camera1[0]*(int)h_camera1[1];

    printf("Allocating %d x %d pixels for xray 1\n",(int)h_camera1[0],(int)h_camera1[1]);

    free(h_xray1);
    h_xray1 = (float*)malloc(numpx*sizeof(float));

    free(h_xrayGx1);
    h_xrayGx1 = (float*)malloc(numpx*sizeof(float));

    free(h_xrayGy1);
    h_xrayGy1 = (float*)malloc(numpx*sizeof(float));

    free(h_DRR1);
    h_DRR1 = (float*)malloc(numpx*sizeof(float));

    free(h_DRRGx1);
    h_DRRGx1 = (float*)malloc(numpx*sizeof(float));

    free(h_DRRGy1);
    h_DRRGy1 = (float*)malloc(numpx*sizeof(float));

    cudaFree(d_xray1);
    cudaStatus = cudaMalloc((void **) &d_xray1,   numpx*sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the xray failed!\n");return;}

    cudaFree(d_xrayGx1);
    cudaStatus = cudaMalloc((void **) &d_xrayGx1,   numpx*sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the xray Gx failed!\n");return;}

    cudaFree(d_xrayGy1);
    cudaStatus = cudaMalloc((void **) &d_xrayGy1,   numpx*sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the xray Gy failed!\n");return;}

    cudaFree(d_DRR1);
    cudaStatus = cudaMalloc((void **) &d_DRR1,   numpx*sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the DRR failed!\n");return;}

    cudaFree(d_DRRGx1);
    cudaStatus = cudaMalloc((void **) &d_DRRGx1,   numpx*sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the DRR Gx failed!\n");return;}

    cudaFree(d_DRRGy1);
    cudaStatus = cudaMalloc((void **) &d_DRRGy1,   numpx*sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the DRR Gy failed!\n");return;}

}

void RayCast::AllocateXray2()
{
    int numpx = (int)h_camera2[0]*(int)h_camera2[1];

    printf("Allocating %d x %d pixels for xray 2\n",(int)h_camera2[0],(int)h_camera2[1]);

    free(h_xray2);
    h_xray2 = (float*)malloc(numpx*sizeof(float));

    free(h_xrayGx2);
    h_xrayGx2 = (float*)malloc(numpx*sizeof(float));

    free(h_xrayGy2);
    h_xrayGy2 = (float*)malloc(numpx*sizeof(float));

    free(h_DRR2);
    h_DRR2 = (float*)malloc(numpx*sizeof(float));

    free(h_DRRGx2);
    h_DRRGx2 = (float*)malloc(numpx*sizeof(float));

    free(h_DRRGy2);
    h_DRRGy2 = (float*)malloc(numpx*sizeof(float));

    cudaFree(d_xray2);
    cudaStatus = cudaMalloc((void **) &d_xray2,   numpx*sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the xray failed!\n");return;}

    cudaFree(d_xrayGx2);
    cudaStatus = cudaMalloc((void **) &d_xrayGx2,   numpx*sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the xray Gx failed!\n");return;}

    cudaFree(d_xrayGy2);
    cudaStatus = cudaMalloc((void **) &d_xrayGy2,   numpx*sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the xray Gy failed!\n");return;}

    cudaFree(d_DRR2);
    cudaStatus = cudaMalloc((void **) &d_DRR2,   numpx*sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the DRR failed!\n");return;}

    cudaFree(d_DRRGx2);
    cudaStatus = cudaMalloc((void **) &d_DRRGx2,   numpx*sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the DRR Gx failed!\n");return;}

    cudaFree(d_DRRGy2);
    cudaStatus = cudaMalloc((void **) &d_DRRGy2,   numpx*sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the DRR Gy failed!\n");return;}

}

void RayCast::AllocateCT()
{
    free(h_CT);
    int w = (int)h_ctParams[0];
    int h = (int)h_ctParams[1];
    int d = (int)h_ctParams[2];
    int numvox = w*h*d;
    h_CT = (float*)malloc(numvox*sizeof(float));
    volSize = make_cudaExtent(w,h,d);
    cudaFreeArray(d_CT);
    cudaMalloc3DArray(&d_CT, &channelDescCT, volSize);

}

void RayCast::CreateCTBoundingBox()
{
    int x = (int)h_ctParams[0];
    int y = (int)h_ctParams[1];
    int z = (int)h_ctParams[2];

    h_BoxCT[ 0]=0; h_BoxCT[ 1]=x; h_BoxCT[ 2]=0; h_BoxCT[ 3]=x;h_BoxCT[ 4]=0; h_BoxCT[ 5]=x; h_BoxCT[ 6]=0; h_BoxCT[ 7]=x;
    h_BoxCT[ 8]=0; h_BoxCT[ 9]=0; h_BoxCT[10]=y; h_BoxCT[11]=y;h_BoxCT[12]=0; h_BoxCT[13]=0; h_BoxCT[14]=y; h_BoxCT[15]=y;
    h_BoxCT[16]=0; h_BoxCT[17]=0; h_BoxCT[18]=0; h_BoxCT[19]=0;h_BoxCT[20]=z; h_BoxCT[21]=z; h_BoxCT[22]=z; h_BoxCT[23]=z;
    h_BoxCT[24]=1; h_BoxCT[25]=1; h_BoxCT[26]=1; h_BoxCT[27]=1;h_BoxCT[28]=1; h_BoxCT[29]=1; h_BoxCT[30]=1; h_BoxCT[31]=1;
}

void RayCast::PrintBoundingBox(int num)
{
    if(num==1)
    {
        printf("Bounding Box Left %d W %d Top %d H %d\n", h_BoundingBox1[0],h_BoundingBox1[1],h_BoundingBox1[2],h_BoundingBox1[3]);
    }
    else if(num==2)
    {
        printf("Bounding Box Left %d W %d Top %d H %d\n", h_BoundingBox2[0],h_BoundingBox2[1],h_BoundingBox2[2],h_BoundingBox2[3]);
    }

}

void RayCast::ComputeDRRGradient1()
{
    CudaEntryFilterImage1(d_DRR1,d_DRRGx1,d_DRRGy1);
}

void RayCast::ComputeDRRGradient2()
{
    CudaEntryFilterImage2(d_DRR2,d_DRRGx2,d_DRRGy2);
}

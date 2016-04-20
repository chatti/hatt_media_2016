#include "DASH.h"
#include <sstream>

cudaError_t EntryCostFunctionDASH_MonoPlane(float* d_pc, float* d_M, float* d_cost, float* h_cost, float* finalCost, int B, int T);
cudaError_t EntryCostFunctionGDASH_MonoPlane(float* d_pc, float* d_M, float* d_cost, float* h_cost, float* finalCost, int B, int T);
cudaError_t EntryCostFunctionDASH_BiPlane(float* d_pc, float* d_M1, float* d_M2, float* d_cost, float* h_cost, float* finalCost, int B, int T);
cudaError_t EntrySetupTextureDASH_X1(cudaArray* dI, cudaChannelFormatDesc chDesc);
cudaError_t EntrySetupTextureDASH_Gx1(cudaArray* dI, cudaChannelFormatDesc chDesc);
cudaError_t EntrySetupTextureDASH_Gy1(cudaArray* dI, cudaChannelFormatDesc chDesc);
cudaError_t EntrySetupTextureDASH_X2(cudaArray* dI, cudaChannelFormatDesc chDesc);
cudaError_t EntrySetConstantDataDASH(float* camera1, float* camera2, float pcsize);
cudaError_t EntryTextureTest(float *dpc, float* dT, int B, int T);

//cudaError_t EntryCostFunctionDASH2(float* dPC, float* phi, float* dcost, int N, float* cost, float* finalCost);
//cudaError_t EntrySetupTextureDASH(cudaArray* dI, cudaChannelFormatDesc chDesc);
//cudaError_t EntrySetConstantDataDASH(float* camera, float* pcParams, float* phi);

DASH::DASH(){
	
	cpuStatus=0;
	//Reset and Initialize the GPU
	StartUpGPU();
	if(cudaStatus != cudaSuccess){printf("Problem Starting up the GPU!\n");return;}

    metric=1;
	//Initialize everything to just a single float
	h_xray1    = (float*)malloc(sizeof(float));
	h_xray2    = (float*)malloc(sizeof(float));
    h_xrayGx1    = (float*)malloc(sizeof(float));
    h_xrayGx2    = (float*)malloc(sizeof(float));
    h_xrayGy1    = (float*)malloc(sizeof(float));
    h_xrayGy2    = (float*)malloc(sizeof(float));
	h_pcloud   = (float*)malloc(sizeof(float));
	h_gcloud   = (float*)malloc(sizeof(float));
    h_cost     = (float*)malloc(sizeof(float));
	//Now move onto the GPU memory.  We know how big this needs to be (4x4 of floats)
	cudaStatus = cudaMalloc((void **) &d_M1,   16*sizeof(float));
	if(cudaStatus!=cudaSuccess){printf("Allocating the transform parameters failed!\n");return;}
	cudaStatus = cudaMalloc((void **) &d_M2,   16*sizeof(float));
	if(cudaStatus!=cudaSuccess){printf("Allocating the transform parameters failed!\n");return;}

	//Just initialize this to 1	
	cudaStatus = cudaMalloc((void **) &d_cost,  sizeof(float));
	if(cudaStatus!=cudaSuccess){printf("Allocating the cost function failed!\n");return;}
	cudaStatus = cudaMalloc((void **) &d_pcloud,   sizeof(float));
	if(cudaStatus!=cudaSuccess){printf("Allocating the transform parameters failed!\n");return;}
    cudaStatus = cudaMalloc((void **) &d_gcloud,   sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the transform parameters failed!\n");return;}

	channelDescXray = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	//Copy the X-ray memory to the GPU. These are textures
	cudaStatus = cudaMallocArray(&d_xray1, &channelDescXray,1,1);
	if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray image failed!\n");return;}
	cudaStatus = cudaMallocArray(&d_xrayGx1, &channelDescXray,1,1);
	if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray x-gradient image failed!\n");return;}
	cudaStatus = cudaMallocArray(&d_xrayGy1, &channelDescXray,1,1);
	if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray y-gradient image failed!\n");return;}

	//Copy the X-ray memory to the GPU. These are textures
	cudaStatus = cudaMallocArray(&d_xray2, &channelDescXray,1,1);
	if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray image failed!\n");return;}
	cudaStatus = cudaMallocArray(&d_xrayGx2, &channelDescXray,1,1);
	if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray x-gradient image failed!\n");return;}
	cudaStatus = cudaMallocArray(&d_xrayGy2, &channelDescXray,1,1);
	if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray y-gradient image failed!\n");return;}

    T=512;

}

DASH::~DASH(){
	free(h_xray1);
	free(h_xray2);
    free(h_xrayGx1);
    free(h_xrayGx2);
    free(h_xrayGy1);
    free(h_xrayGy2);
	free(h_cost);
	free(h_pcloud);
	free(h_gcloud);
    cudaFreeArray(d_xray1);
    cudaFreeArray(d_xrayGx1);
    cudaFreeArray(d_xrayGy1);
    cudaFreeArray(d_xray2);
    cudaFreeArray(d_xrayGx2);
    cudaFreeArray(d_xrayGy2);
	cudaFree(d_pcloud);
	cudaFree(d_gcloud);
	cudaFree(d_cost);
    cudaFree(d_M1);
    cudaFree(d_M2);
}

int DASH::GetNumXray()
{
    return numXray;
}

void DASH::SetMetric(int m)
{
    metric=m;
}

float DASH::ComputeDASHCostFunction()
{

    if(metric==1)
    {
        if(numXray==1)
        {

            cudaStatus = EntryCostFunctionDASH_MonoPlane(d_pcloud, d_M1, d_cost, h_cost, h_finalCost, B, T);
            return h_finalCost[0];
        }
        else if(numXray==2)
        {
            //Not enabled yet
            //cudaStatus = EntryCostFunctionDASH_BiPlane(d_pcloud, d_M1,d_M2, d_cost, h_cost, h_finalCost, B, T);
            //return h_finalCost[0];
        }
        else
        {
            printf("Cost function number of X-rays not set correctly\n");
            cpuStatus=-3;
            return 9999;
        }
    }
    else
    {
        if(numXray==1)
        {
            cudaStatus = EntryCostFunctionGDASH_MonoPlane(d_pcloud, d_M1, d_cost, h_cost, h_finalCost, B, T);
            return h_finalCost[0];
        }
        else if(numXray==2)
        {
            //Not enabled yet
            //cudaStatus = EntryCostFunctionDASH_BiPlane(d_pcloud, d_M1,d_M2, d_cost, h_cost, h_finalCost, B, T);
            //return h_finalCost[0];
        }
        else
        {
            printf("Cost function number of X-rays not set correctly\n");
            cpuStatus=-3;
            return 9999;
        }
    }

}


void DASH::StartUpGPU()
{
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess){printf("InitGPU: Setting the Device to 0 failed!\n");return;}
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess){printf("InitGPU: Resetting the Device to 0 failed!\n");return;}
}

void DASH::SetCamera1Params(float f, float W, float H, float sx, float sy, float iso, float* phi)
{	
	h_camera1[0]=W; h_camera1[1]=H;
	h_camera1[2]=sx;h_camera1[3]=sy;
	h_camera1[4]=f; h_camera1[5]=iso;
	for(int k=0; k<6; k++)
	{
		h_camera1[6+k]=phi[k];
	}
}

void DASH::SetCamera2Params(float f, float W, float H, float sx, float sy, float iso, float* phi)
{	
	h_camera2[0]=W; h_camera2[1]=H;
	h_camera2[2]=sx;h_camera2[3]=sy;
	h_camera2[4]=f; h_camera2[5]=iso;
	for(int k=0; k<6; k++)
	{
		h_camera2[6+k]=phi[k];
	}
}

void DASH::SetTransform(float tx, float ty, float tz, float rx, float ry, float rz)
{
    h_transform[0]=tx;
    h_transform[1]=ty;
    h_transform[2]=tz;
    h_transform[3]=rx;
    h_transform[4]=ry;
    h_transform[5]=rz;
}

void DASH::SetInitialTransform(float tx, float ty, float tz, float rx, float ry, float rz)
{
    h_init[0]=tx;
    h_init[1]=ty;
    h_init[2]=tz;
    h_init[3]=rx;
    h_init[4]=ry;
    h_init[5]=rz;
}

void DASH::SetPCloudParams(float N)
{
	pcloudSize = N;
}

void DASH::ReadXray1(std::string filename){
    int W = (int)h_camera1[0];
    int H = (int)h_camera1[1];
    free(h_xray1);
    h_xray1 = (float*)malloc(W*H*sizeof(float));
    ReadImage(filename, h_xray1, W, H);
	TransferXray1ToGPU();
}

void DASH::ReadXrayGx1(std::string filename){
    int W = (int)h_camera1[0];
    int H = (int)h_camera1[1];
    free(h_xrayGx1);
    h_xrayGx1 = (float*)malloc(W*H*sizeof(float));
    ReadImage(filename, h_xrayGx1, W, H);
    TransferXray1ToGPU();
}

void DASH::ReadXrayGy1(std::string filename){
    int W = (int)h_camera1[0];
    int H = (int)h_camera1[1];
    free(h_xrayGy1);
    h_xrayGy1 = (float*)malloc(W*H*sizeof(float));
    ReadImage(filename, h_xrayGy1, W, H);
    TransferXray1ToGPU();
}

void DASH::ReadXray2(std::string filename){
    int W = (int)h_camera2[0];
    int H = (int)h_camera2[1];
    free(h_xray2);
    h_xray2 = (float*)malloc(W*H*sizeof(float));
    ReadImage(filename, h_xray2, W, H);
	TransferXray2ToGPU();
}

void DASH::ReadXrayGx2(std::string filename){
    int W = (int)h_camera2[0];
    int H = (int)h_camera2[1];
    free(h_xrayGx2);
    h_xrayGx2 = (float*)malloc(W*H*sizeof(float));
    ReadImage(filename, h_xrayGx2, W, H);
    TransferXray2ToGPU();
}

void DASH::ReadXrayGy2(std::string filename){
    int W = (int)h_camera2[0];
    int H = (int)h_camera2[1];
    free(h_xrayGy2);
    h_xrayGy2 = (float*)malloc(W*H*sizeof(float));
    ReadImage(filename, h_xrayGy2, W, H);
    TransferXray2ToGPU();
}

void DASH::ReadPCloud(std::string filename){
    free(h_pcloud);
    int N = (int)pcloudSize;
    h_pcloud = (float*)malloc(7*N*sizeof(float));
    cudaFree(d_pcloud);
    cudaStatus = cudaMalloc((void **) &d_pcloud,   7*N*sizeof(float));
    if (cudaStatus != cudaSuccess){printf("Allocating the point cloud to the GPU failed!\n");}

    ReadImage(filename, h_pcloud, N, 7);

    B = N/T;
    free(h_cost);
    cudaFree(d_cost);
    h_cost = (float*)malloc(B*sizeof(float));
    cudaStatus = cudaMalloc((void **) &d_cost,   B*sizeof(float));
    if (cudaStatus != cudaSuccess){printf("Allocating the cost function to the GPU failed!\n");}

	TransferPCloudToGPU();
}

void DASH::ReadGCloud(std::string filename){
    free(h_gcloud);
    h_gcloud = (float*)malloc(3*(int)pcloudSize*sizeof(float));
    ReadImage(filename, h_gcloud, 3*(int)pcloudSize, 1.0);
	TransferGCloudToGPU();
}

void DASH::TransferXray1ToGPU(){
	
	int W=(int)h_camera1[0];
	int H=(int)h_camera1[1];

	cudaFreeArray(d_xray1);
    cudaFreeArray(d_xrayGx1);
    cudaFreeArray(d_xrayGy1);

	cudaStatus = cudaMallocArray(&d_xray1, &channelDescXray,W,H);
	if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray image failed!\n");return;}
    cudaStatus = cudaMallocArray(&d_xrayGx1, &channelDescXray,W,H);
    if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray x-gradient image failed!\n");return;}
    cudaStatus = cudaMallocArray(&d_xrayGy1, &channelDescXray,W,H);
    if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray y-gradient image failed!\n");return;}

    cudaStatus = EntrySetupTextureDASH_X1(d_xray1, channelDescXray);
    if (cudaStatus != cudaSuccess){printf("Binding the X-ray image 1 to the GPU failed!\n");}

	cudaStatus = cudaMemcpyToArray(d_xray1, 0,0, h_xray1, W*H*sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess){printf("Transfering the X-ray image 1 to the GPU failed!\n");}

    cudaStatus = EntrySetupTextureDASH_Gx1(d_xrayGx1, channelDescXray);
    if (cudaStatus != cudaSuccess){printf("Binding the X-ray image 1 to the GPU failed!\n");}

    cudaStatus = cudaMemcpyToArray(d_xrayGx1, 0,0, h_xrayGx1, W*H*sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess){printf("Transfering the X-ray image 1 to the GPU failed!\n");}

    cudaStatus = EntrySetupTextureDASH_Gy1(d_xrayGy1, channelDescXray);
    if (cudaStatus != cudaSuccess){printf("Binding the X-ray image 1 to the GPU failed!\n");}

    cudaStatus = cudaMemcpyToArray(d_xrayGy1, 0,0, h_xrayGy1, W*H*sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess){printf("Transfering the X-ray image 1 to the GPU failed!\n");}
	
}

void DASH::TransferXray2ToGPU(){
	
	int W=(int)h_camera2[0];
	int H=(int)h_camera2[1];
	cudaFreeArray(d_xray2);
	//cudaFreeArray(d_xrayGx1);
	//cudaFreeArray(d_xrayGy1);
	cudaStatus = cudaMallocArray(&d_xray2, &channelDescXray,W,H);
	if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray image failed!\n");return;}
	//cudaStatus = cudaMallocArray(&d_xrayGx1, &channelDescXray,W,H);
	//if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray x-gradient image failed!\n");return;}
	//cudaStatus = cudaMallocArray(&d_xrayGy1, &channelDescXray,W,H);
	//if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray y-gradient image failed!\n");return;}

    cudaStatus = EntrySetupTextureDASH_X2(d_xray2, channelDescXray);
    if (cudaStatus != cudaSuccess){printf("Binding the X-ray image 2 to the GPU failed!\n");}

	cudaStatus = cudaMemcpyToArray(d_xray2, 0,0, h_xray2, W*H*sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess){printf("Transfering the X-ray image 2 to the GPU failed!\n");}
	
}

void DASH::TransferConstantsToGPU()
{
	cudaStatus = EntrySetConstantDataDASH(h_camera1, h_camera2, pcloudSize);
}

void DASH::TransferPCloudToGPU()
{
	int N = (int)pcloudSize;
    cudaStatus = cudaMemcpy(d_pcloud, h_pcloud, 7*N*sizeof(float), cudaMemcpyHostToDevice);
	if(cudaStatus!=cudaSuccess){printf("Transfering the point cloud to the device failed!\n");return;}
}

void DASH::TransferGCloudToGPU()
{
    //Disabled
    //int N = (int)pcloudSize;
    //cudaStatus = cudaMemcpy(d_gcloud, h_gcloud, 3*N*sizeof(float), cudaMemcpyHostToDevice);
    //if(cudaStatus!=cudaSuccess){printf("Transfering the gradient cloud to the device failed!\n");return;}
}

bool DASH::isOK(){
	if(cudaStatus == cudaSuccess & cpuStatus==0)
	{return true;}
	else if(cudaStatus != cudaSuccess)
	{
		printf("CUDA Error code: %d\n",cudaStatus);
		return false;
	}
	else
	{
		printf("CPU code: %d\n",cudaStatus);
		return false;
	}
}

//This function works on row-major 2d matrices stored as 1D arrays
void DASH::MM44(float (&MO)[16], float (&ML)[16], float (&MR)[16])
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

void DASH::PrintM44(float (&M)[16])
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

void DASH::SetEye(float (&M)[16])
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

void DASH::SetSpatialTranslation(float (&M)[16],float* tform)
{
    M[3]=tform[0];
    M[7]=tform[1];
    M[11]=tform[2];
}

void DASH::SetSpatialRotation(float (&M)[16],float* tform)
{
    float cx = cos(tform[3]);float sx = sin(tform[3]);
    float cy = cos(-tform[4]);float sy = sin(-tform[4]);
    float cz = cos(tform[5]);float sz = sin(tform[5]);

    M[0] = cy*cz+sx*sy*sz; M[1] = -cx*sz; M[2] = cy*sx*sz-cz*sy;
    M[4] = cy*sz-cz*sx*sy; M[5] =  cx*cz; M[6] = -sy*sz-cy*cz*sx;
    M[8] = cx*sy;          M[9] =  sx;    M[10] = cx*cy;
}

void DASH::UpdateTransformMatrix()
{
    SetSpatialTranslation(h_T,h_transform);
    SetSpatialRotation(h_R,h_transform);

    //The full spatial transformation is decomposed into probe position and camera position
    //Probe position = F = T*iT*R*iR
    //Camera position = C = cT*(iso-1)*cR*iso;
    MM44(TMP1,   h_R,        h_iR);
    MM44(TMP2,   h_iT,       TMP1);
    MM44(h_F,    h_T,        TMP2);
    MM44(TMP2,   h_C1,   h_F);


    cudaStatus = cudaMemcpy(d_M1, TMP2, 16*sizeof(float), cudaMemcpyHostToDevice);
    if(cudaStatus!=cudaSuccess){printf("Transfering transform matrix 1 failed!\n");return;}
    if(numXray==2)
	{
        MM44(TMP2,   h_C2,      h_F);
        cudaStatus = cudaMemcpy(d_M2,TMP2 , 16*sizeof(float), cudaMemcpyHostToDevice);
        if(cudaStatus!=cudaSuccess){printf("Transfering transform matrix 2 failed!\n");return;}
	}

}


void DASH::InitAllMatrices()
{
    float cR[16];
    float cT[16];
    float iso[16];
    SetEye(h_T);
    SetEye(h_R);
    SetEye(h_iT);
    SetEye(h_iR);
    SetEye(cR);
    SetEye(cT);
    SetEye(iso);

    SetSpatialTranslation(h_iT,h_init);
    SetSpatialRotation(h_iR,h_init);

    float tmp[6];
    for( int k=0; k<6; k++){
        tmp[k] = -h_camera1[k+6];
    }
    SetSpatialTranslation(cT,tmp);
    SetSpatialRotation(cR,tmp);

    //This matrix moves the probe to its position relative to isocenter of the c-arm
    iso[11]=-h_camera1[5];


    MM44(TMP1,cR,iso);
    iso[11]=-iso[11]; //Now move away from isocenter back to the camera detector
    MM44(TMP2,iso,TMP1);
    MM44(h_C1,cT,TMP2);

    //Now do camera 2
    for( int k=0; k<6; k++){
        tmp[k] = -h_camera2[k+6];
    }
    SetSpatialTranslation(cT,tmp);
    SetSpatialRotation(cR,tmp);

    //This matrix moves the probe to its position relative to isocenter of the c-arm
    iso[11]=-h_camera2[5];

    MM44(TMP1,cR,iso);

    iso[11]=-iso[11]; //Now move away from isocenter back to the camera detector
    MM44(TMP2,iso,TMP1);
    MM44(h_C2,cT,TMP2);


}

void DASH::ReadParameterFile(std::string filename)
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
        else if(param == "camera1")
        {

            printf("Camera 1 params: ");
            for(int k=0;k<12;k++)
            {
                ss >> h_camera1[k];
                printf("%f\t",h_camera1[k]);
            }
            printf("\n");
        }
        else if(param == "camera2")
        {
            printf("Camera 2 params: ");
            for(int k=0;k<12;k++)
            {
                ss >> h_camera2[k];
                printf("%f\t",h_camera2[k]);

            }
            printf("\n");
        }

        else if(param == "xrayfilename1")
        {
            ss >> str;
            ReadXray1(str.c_str());
            printf("X-ray filename 1: %s\n",str.c_str());
        }

        else if(param == "xrayGxfilename1")
        {
            ss >> str;
            ReadXrayGx1(str.c_str());
            printf("X-ray Gx filename 1: %s\n",str.c_str());
        }

        else if(param == "xrayGyfilename1")
        {
            ss >> str;
            ReadXrayGy1(str.c_str());
            printf("X-ray Gy filename 1: %s\n",str.c_str());
        }

        else if(param == "xrayfilename2")
        {
            ss >> str;
            ReadXray2(str.c_str());
            printf("X-ray filename 2: %s\n",str.c_str());
        }

        else if(param == "pointcloudsize")
        {
            ss >> pcloudSize;
            printf("Point cloud size: %d\n",(int)pcloudSize);
        }

        else if(param == "pointcloudfilename")
        {
            ss >> str;
            ReadPCloud(str.c_str());
            printf("Point cloud file name: %s\n",str.c_str());
        }
        else if(param == "gradientcloudfilename")
        {
            ss >> str;
            ReadGCloud(str.c_str());
            printf("Gradient cloud file name: %s\n",str.c_str());
        }

        else if(param == "initialtransform")
        {
            printf("Initial transform: ");
            for(int k=0;k<6;k++)
            {
                ss >> h_init[k];
                h_transform[k]=0.0;
                printf("%f\t",h_init[k]);
            }
            printf("\n");
        }

        count++;
    }
    infile.close();
    InitAllMatrices();
    TransferConstantsToGPU();
}

void DASH::ReadImage(std::string filename, float* img, int W, int H)
{
    FILE *fp;
    fp = fopen(filename.c_str(), "r");
    if(fp==NULL){
        cpuStatus = -1;
        return;
    }
    size_t r = fread(img, sizeof(float), W*H, fp);
    fclose(fp);
    return;
}


void DASH::WriteVolume(std::string filename, float* vol, int sx, int sy, int sz)
{
    FILE *fp;
    fp = fopen(filename.c_str(), "w");
    if(fp==NULL){
        cpuStatus = -2;
        return;
    }
    size_t r = fwrite(vol, sizeof(float), sx*sy*sz, fp);
    fclose(fp);
    return;
}

void DASH::WriteImage(std::string filename, float* img, int W, int H)
{
    FILE *fp;
    fp = fopen(filename.c_str(), "w");
    if(fp==NULL){
        cpuStatus = -2;
        return;
    }
    size_t r = fwrite(img, sizeof(float), W*H, fp);
    fclose(fp);
    return;
}

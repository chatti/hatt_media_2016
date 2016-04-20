#include "PatchRayCast.h"


cudaError_t CudaEntryBindCT(cudaArray* d_CT, cudaChannelFormatDesc chDescCT);
cudaError_t EntrySetupTextureXrayGx1(cudaArray* d_xrayGx, cudaChannelFormatDesc chDesc);
cudaError_t EntrySetupTextureXrayGx2(cudaArray* d_xrayGy, cudaChannelFormatDesc chDesc);
cudaError_t EntrySetupTextureXrayGy1(cudaArray* d_xrayGx, cudaChannelFormatDesc chDesc);
cudaError_t EntrySetupTextureXrayGy2(cudaArray* d_xrayGy, cudaChannelFormatDesc chDesc);
cudaError_t CudaEntryPatchGCC(float* h_Cam2CT, float* h_rayParams, float* h_patchLocations, float* h_pGCC, float* d_pGCC);
cudaError_t CudaEntrySetConstantData1(float* camera);
cudaError_t CudaEntrySetConstantData2(float* camera);

cudaError_t CudaEntryPatchGCCTest(float* h_Cam2CT, float* h_rayParams, float* h_patchLocations, float* h_pGCC, float* d_pGCC);

PatchRayCast::PatchRayCast()
{

    cpuStatus=0;
    //Initialize the GPU
    //StartUpGPU();
    //if(cudaStatus != cudaSuccess){printf("Problem Starting Up the GPU!\n");return;}

    //Initialize CT
    h_CT          = (float*)malloc(sizeof(float));

    //Initialize X-ray and DRRs
    h_xray1       = (float*)malloc(sizeof(float));
    h_xray2       = (float*)malloc(sizeof(float));
    h_xrayGx1     = (float*)malloc(sizeof(float));
    h_xrayGx2     = (float*)malloc(sizeof(float));
    h_xrayGy1     = (float*)malloc(sizeof(float));
    h_xrayGy2     = (float*)malloc(sizeof(float));

    channelDescImage = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    cudaStatus = cudaMallocArray(&d_xrayGx1, &channelDescImage,1,1);
    if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray Gx 1 image failed!\n");return;}
    cudaStatus = cudaMallocArray(&d_xrayGy1, &channelDescImage,1,1);
    if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray Gy 1 image failed!\n");return;}
    cudaStatus = cudaMallocArray(&d_xrayGx2, &channelDescImage,1,1);
    if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray Gx 2 image failed!\n");return;}
    cudaStatus = cudaMallocArray(&d_xrayGy2, &channelDescImage,1,1);
    if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray Gy 2 image failed!\n");return;}

    //Initialize the CT
    channelDescCT = cudaCreateChannelDesc<float>();
    volSize = make_cudaExtent(1, 1, 1);
    volSize.width  = 1;
    volSize.height = 1;
    volSize.depth  = 1;
    cudaStatus=cudaMalloc3DArray(&d_CT, &channelDescCT, volSize);
    if(cudaStatus!=cudaSuccess){printf("Allocating the CT failed!\n");return;}

    cudaStatus = cudaMalloc((void **) &d_PGCC, 2*PATCHCOUNT*sizeof(float));
    if(cudaStatus!=cudaSuccess){printf("Allocating the cost function parameters failed!\n");return;}

}

PatchRayCast::~PatchRayCast()
{
    //Host
    free(h_xray1);
    free(h_xray2);
    free(h_xrayGx1);
    free(h_xrayGx2);
    free(h_xrayGy1);
    free(h_xrayGy2);
    free(h_CT);

    //Device
    cudaFreeArray(d_xrayGx1);
    cudaFreeArray(d_xrayGy1);
    cudaFreeArray(d_xrayGx2);
    cudaFreeArray(d_xrayGy2);
    cudaFreeArray(d_CT);
    cudaFree(d_PGCC);
}

int PatchRayCast::GetNumXray()
{
    return numXray;
}


float PatchRayCast::ComputePatchGCC()
{
    UpdateTransformMatrix();
    UpdatePatchLocations1();
    cudaStatus = CudaEntryPatchGCC(h_Cam2CT1, h_rayParams1, h_PatchLocations1, h_PGCC, d_PGCC);
    //cudaStatus = CudaEntryPatchGCCTest(h_Cam2CT1, h_rayParams1, h_PatchLocations1, h_PGCC, d_PGCC);

    PGCC = 0.0;
    for(int k=0; k< 2*PATCHCOUNT;k++)
    {
        //printf("%d - %f, ",k,h_PGCC[k]);
        PGCC = PGCC + h_PGCC[k];
    }
    //printf("\n");
    PGCC = -PGCC/(2*PATCHCOUNT);

    return PGCC;
}

void PatchRayCast::StartUpGPU()
{
	cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        printf("InitGPU: Setting the Device to 0 failed!\n");
		return;
    }

	cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        printf("InitGPU: Resetting the Device to 0 failed!\n");
		return;
    }
    if (cudaStatus == cudaSuccess)
    {
        printf("InitGPU: GPU Started up OK!\n");
    }
}

void PatchRayCast::ReadImage(std::string filename, float* img, int W, int H)
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

void PatchRayCast::ReadVolume(std::string filename, float* vol, int sx, int sy, int sz)
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

void PatchRayCast::WriteVolume(std::string filename, float* vol, int sx, int sy, int sz)
{
	FILE *fp;
    fp = fopen(filename.c_str(), "w");
	if(fp==NULL){
        printf("Error writing: %s\n",filename.c_str());
		cpuStatus = -2;	
		return;
	}
	size_t r = fwrite(vol, sizeof(float), sx*sy*sz, fp);
	fclose(fp);
	return;
}

void PatchRayCast::WriteImage(std::string filename, float* img, int W, int H)
{
	FILE *fp;
    fp = fopen(filename.c_str(), "w");
	if(fp==NULL){
        printf("Error writing: %s\n",filename.c_str());
		cpuStatus = -2;	
		return;
	}
	size_t r = fwrite(img, sizeof(float), W*H, fp);
	fclose(fp);
	return;
}


void PatchRayCast::SetCTParams(float xdim, float ydim, float zdim, float dx, float dy, float dz, float cx, float cy, float cz)
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

void PatchRayCast::ReadXray1(std::string filename, float W, float H)
{
    AllocateXray1();
    ReadImage(filename, h_xray1, (int)W, (int)H);
    ComputeXrayGradient(1);
    TransferXrayGPU(1);
    cudaStatus =  CudaEntrySetConstantData1(h_camera1);
}

void PatchRayCast::ReadXray2(std::string filename, float W, float H)
{
    AllocateXray1();
    ReadImage(filename, h_xray2, (int)W, (int)H);
    ComputeXrayGradient(2);
    TransferXrayGPU(2);
    cudaStatus =  CudaEntrySetConstantData2(h_camera2);
}

void PatchRayCast::ReadCT(std::string filename)
{

    AllocateCT();
    ReadVolume(filename, h_CT, (int)h_ctParams[0], (int)h_ctParams[1], (int)h_ctParams[2]);
	TransferCTGPU();
}

void PatchRayCast::TransferXrayGPU(int xraynum)
{
    if(xraynum==1)
    {
        int W=(int)h_camera1[0];
        int H=(int)h_camera1[1];
        cudaStatus = EntrySetupTextureXrayGx1(d_xrayGx1, channelDescImage);
        cudaStatus = EntrySetupTextureXrayGy1(d_xrayGy1, channelDescImage);
        cudaStatus = cudaMemcpyToArray(d_xrayGx1, 0,0, h_xrayGx1, W*H*sizeof(float), cudaMemcpyHostToDevice);
        cudaStatus = cudaMemcpyToArray(d_xrayGy1, 0,0, h_xrayGy1, W*H*sizeof(float), cudaMemcpyHostToDevice);
    }
    else
    {
        int W=(int)h_camera2[0];
        int H=(int)h_camera2[1];
        cudaStatus = EntrySetupTextureXrayGx2(d_xrayGx2, channelDescImage);
        cudaStatus = EntrySetupTextureXrayGy2(d_xrayGy2, channelDescImage);
        cudaStatus = cudaMemcpyToArray(d_xrayGx2, 0,0, h_xrayGx2, W*H*sizeof(float), cudaMemcpyHostToDevice);
        cudaStatus = cudaMemcpyToArray(d_xrayGy2, 0,0, h_xrayGy2, W*H*sizeof(float), cudaMemcpyHostToDevice);
    }
}


void PatchRayCast::TransferCTGPU()
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


void PatchRayCast::TransferConstantsToGPU()
{

    cudaStatus =  CudaEntrySetConstantData1(h_camera1);
    if(cudaStatus!=cudaSuccess)
    {
        return;
    }
    cudaStatus =  CudaEntrySetConstantData2(h_camera2);
    if(cudaStatus!=cudaSuccess)
    {
        return;
    }

}

void PatchRayCast::WriteXrayGx(std::string filename, int num){

    if(num==1)
    {
        WriteImage(filename,h_xrayGx1, (int)h_camera1[0], (int)h_camera1[1]);
    }
    else
    {
        WriteImage(filename,h_xrayGx2, (int)h_camera2[0], (int)h_camera2[1]);
    }
}

void PatchRayCast::WriteXrayGy(std::string filename, int num){

    if( num==1)
    {
        WriteImage(filename,h_xrayGy1, (int)h_camera1[0], (int)h_camera1[1]);
    }
    else
    {
        WriteImage(filename,h_xrayGy2, (int)h_camera2[0], (int)h_camera2[1]);
    }
}

bool PatchRayCast::isOK(){
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

void PatchRayCast::SetInitialTransform(float tx, float ty, float tz, float rx, float ry, float rz)
{

    h_initial[0] = tx;
    h_initial[1] = ty;
    h_initial[2] = tz;
    h_initial[3] = rx;
    h_initial[4] = ry;
    h_initial[5] = rz;

}

void PatchRayCast::SetSecondaryTransform(float tx, float ty, float tz, float rx, float ry, float rz)
{
    h_secondary[0] = tx;
    h_secondary[1] = ty;
    h_secondary[2] = tz;
    h_secondary[3] = rx;
    h_secondary[4] = ry;
    h_secondary[5] = rz;
}

void PatchRayCast::SetSpatialTranslation(float (&M)[16],float* tform)
{
    M[3]=tform[0];
    M[7]=tform[1];
    M[11]=tform[2];
}

void PatchRayCast::SetSpatialTranslationInverse(float (&M)[16],float* tform)
{
    M[3]=-tform[0];
    M[7]=-tform[1];
    M[11]=-tform[2];
}

void PatchRayCast::SetSpatialRotation(float (&M)[16],float* tform)
{
    float cx = cos(tform[3]);float sx = sin(tform[3]);
    float cy = cos(-tform[4]);float sy = sin(-tform[4]);
    float cz = cos(tform[5]);float sz = sin(tform[5]);

    M[0] = cy*cz+sx*sy*sz; M[1] = -cx*sz; M[2] = cy*sx*sz-cz*sy;
    M[4] = cy*sz-cz*sx*sy; M[5] =  cx*cz; M[6] = -sy*sz-cy*cz*sx;
    M[8] = cx*sy;          M[9] =  sx;    M[10] = cx*cy;

}

void PatchRayCast::SetSpatialRotationInverse(float (&M)[16],float* tform)
{
    float cx = cos(tform[3]);float sx = sin(tform[3]);
    float cy = cos(-tform[4]);float sy = sin(-tform[4]);
    float cz = cos(tform[5]);float sz = sin(tform[5]);

    M[0] = cy*cz+sx*sy*sz; M[1] = cy*sz-cz*sx*sy;  M[2] = cx*sy;
    M[4] = -cx*sz;         M[5] = cx*cz;           M[6] = sx;
    M[8] = cy*sx*sz-cz*sy; M[9] = -sy*sz-cy*cz*sx; M[10] = cx*cy;

}

void PatchRayCast::UpdateTransformMatrix()
{

    float sT[16]; SetEye(sT);
    float sR[16]; SetEye(sR);
    float sTinv[16]; SetEye(sTinv);
    float sRinv[16]; SetEye(sRinv);

    SetSpatialTranslation(sT,h_secondary);
    SetSpatialRotation(sR,h_secondary);
    SetSpatialTranslationInverse(sTinv,h_secondary);
    SetSpatialRotationInverse(sRinv,h_secondary);


    MM4_4(TMP1,   sR,        h_iR);
    MM4_4(TMP2,    h_iT,        TMP1);
    MM4_4(h_P2W,  sT, TMP2);
    MM4_4(TMP1, h_P2W, h_CT2P);
    MM4_4(h_CT2Cam1, h_W2Cam1, TMP1);


    MM4_4(TMP1,   h_iTinv,        sTinv);
    MM4_4(TMP2,    sRinv,        TMP1);
    MM4_4(h_W2P,  h_iRinv, TMP2);
    MM4_4(TMP1, h_W2P, h_Cam2W1);
    MM4_4(h_Cam2CT1, h_P2CT, TMP1);



    if(numXray==2)
    {
        MM4_4(TMP1,   sR,        h_iR);
        MM4_4(TMP2,    h_iT,        TMP1);
        MM4_4(h_P2W,  sT, TMP2);
        MM4_4(TMP1, h_P2W, h_CT2P);
        MM4_4(h_CT2Cam2, h_W2Cam2, TMP1);

        MM4_4(TMP1,   h_iTinv,        sTinv);
        MM4_4(TMP2,    sRinv,        TMP1);
        MM4_4(h_W2P,  h_iRinv, TMP2);
        MM4_4(TMP1, h_W2P, h_Cam2W2);
        MM4_4(h_Cam2CT2, h_P2CT, TMP1);
    }

}

void PatchRayCast::MM4_4(float (&O)[16], float (&L)[16], float (&R)[16])
{
    int N = 4; //This is the number of rows in the T matrix, always 4;
    int P = 4; //This is the number of points that are being transformed
    int M = 4; //This is the number of cols in the T matrix, always 4;

    for(int row=0; row<N; row++)
    {
        for(int col=0; col<P; col++)
        {
            O[row*P+col] = 0;
            for(int k=0; k <M; k++)
            {
                O[row*P+col]+=L[row*M+k]*R[k*P+col];
            }
        }
    }
}


void PatchRayCast::MM4_PC(float (&O)[4*PATCHCOUNT], float (&L)[16], float (&R)[4*PATCHCOUNT])
{

    int N = 4; //This is the number of rows in the T matrix, always 4;
    int P = PATCHCOUNT; //This is the number of points that are being transformed
    int M = 4; //This is the number of cols in the T matrix, always 4;

    for(int row=0; row<N; row++)
    {
        for(int col=0; col<P; col++)
        {
            O[row*P+col] = 0;
            for(int k=0; k <M; k++)
            {
                O[row*P+col]+=L[row*M+k]*R[k*P+col];
            }
        }

    }

}

void PatchRayCast::InitForwardMatrices()
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

    MM4_4(h_CT2P,O,S);
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

    MM4_4(TMP1,cT,isoI);
    MM4_4(TMP2,cR,TMP1);
    MM4_4(h_W2Cam1,isoF,TMP2);

    SetEye(h_P1);
    h_P1[0]  = 1/h_camera1[2];
    h_P1[5]  = 1/h_camera1[3];
    h_P1[10] = 0;
    h_P1[14] = -1/h_camera1[4];
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

    MM4_4(TMP1,cT,isoI);
    MM4_4(TMP2,cR,TMP1);
    MM4_4(h_W2Cam2,isoF,TMP2);

    SetEye(h_P2);
    h_P2[0]  = 1/h_camera2[2];
    h_P2[5]  = 1/h_camera2[3];
    h_P2[10] = 0;
    h_P2[14] = -1/h_camera2[4];

}

void PatchRayCast::InitInverseMatrices()
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

    MM4_4(h_P2CT,S,O);
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

    MM4_4(TMP1,cR,isoI);
    MM4_4(TMP2,cT,TMP1);
    MM4_4(h_Cam2W1,isoF,TMP2);

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

    MM4_4(TMP1,cT,isoI);
    MM4_4(TMP2,cR,TMP1);
    MM4_4(h_Cam2W2,isoF,TMP2);
}

void PatchRayCast::SetEye(float (&M)[16])
{
    for(int k=0; k<16; k++)
    {
        M[k]=0.0;
    }
    M[0] =1.0;
    M[5] =1.0;
    M[10]=1.0;
    M[15]=1.0;
}

void PatchRayCast::ReadParameterFile(std::string filename)
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

        else if(param == "keypoints")
        {

            for(int k=0; k<(4*PATCHCOUNT); k++)
            {
                infile >> h_KeyPointsCT[k];
                //printf("%f\n",h_KeyPointsCT[k]);
            }

        }

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess){printf("Problem reading parameter file: \n%s!\n",line.c_str());return;}
        count++;
    }

    infile.close();
    InitForwardMatrices();
    InitInverseMatrices();
}


void PatchRayCast::PrintM44(float (&M)[16])
{
    int cnt=0;
    printf("_____________________________\n");
    for(int i=0; i<4; i++)
    {
        for(int j=0; j<4; j++)
        {
            printf("%.4f\t",M[cnt]);
            cnt++;
        }
        printf("\n");
    }
    printf("_____________________________\n");
}

void PatchRayCast::PrintM4PC(float (&M)[4*PATCHCOUNT])
{
    int cnt=0;
    printf("_____________________________\n");
    for(int i=0; i<4; i++)
    {
        for(int j=0; j<PATCHCOUNT; j++)
        {
            printf("%.2f\t",M[cnt]);
            cnt++;
        }
        printf("\n");
    }
    printf("_____________________________\n");
}


void PatchRayCast::UpdatePatchLocations1()
{


    MM4_PC(A,h_CT2Cam1,h_KeyPointsCT);
    MM4_PC(B,h_P1,A);
    Hom2Screen1(B);

    //This is the z-coordinate of the first transformed keypoint
    float centerz = A[2*PATCHCOUNT];

    //The ray spans 100 mm, which is enough for the probe as long as theta_x is less than 45 degrees
    h_rayParams1[0]   = centerz-50;
    h_rayParams1[1]   = centerz+50;
    h_rayParams1[2]   = ceil(100/h_rayParams1[3]);

}

void PatchRayCast::UpdatePatchLocations2()
{

    MM4_PC(A,h_CT2Cam2,h_KeyPointsCT);
    MM4_PC(B,h_P2,A);
    Hom2Screen2(B);

    //This is the z-coordinate of the first transformed keypoint
    //It represents the center of the CT volume
    //We assume that the largest span of the probe, in mm, in the z-direction
    //is 50 mm;
    float centerz = A[2*PATCHCOUNT];

    h_rayParams2[0]   = centerz-25;
    h_rayParams2[1]   = centerz+25;
    h_rayParams2[2]   = ceil(100/h_rayParams2[3]);
}

void PatchRayCast::Hom2Screen1(float (&M)[4*PATCHCOUNT])
{
    //Convert homogenous
    for(int k=0; k < PATCHCOUNT; k++)
    {
        h_PatchLocations1[k]=M[k]/M[k+3*PATCHCOUNT] + h_camera1[0]/2;
        h_PatchLocations1[k+PATCHCOUNT]=(M[k+PATCHCOUNT]/M[k+3*PATCHCOUNT]  + h_camera1[1]/2);
    }
}

void PatchRayCast::Hom2Screen2(float (&M)[4*PATCHCOUNT])
{
    for(int k=0; k < PATCHCOUNT; k++)
    {
        h_PatchLocations2[k]=h_PatchLocations2[k]/M[k+3*PATCHCOUNT] + h_camera2[0]/2;
        h_PatchLocations2[k+PATCHCOUNT]=h_PatchLocations2[k+PATCHCOUNT]/M[k+3*PATCHCOUNT]  + h_camera2[1]/2;
    }
}

void PatchRayCast::AllocateXray1()
{

    int W=(int)h_camera1[0];
    int H=(int)h_camera1[1];
    int numpx = W*H;

    printf("Allocating %d x %d pixels for xray 1\n",W,H);

    free(h_xray1);
    h_xray1 = (float*)malloc(numpx*sizeof(float));
    free(h_xrayGx1);
    h_xrayGx1 = (float*)malloc(numpx*sizeof(float));
    free(h_xrayGy1);
    h_xrayGy1 = (float*)malloc(numpx*sizeof(float));

    cudaFreeArray(d_xrayGx1);
    cudaFreeArray(d_xrayGy1);

    cudaStatus = cudaMallocArray(&d_xrayGx1, &channelDescImage,W,H);
    if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray device image Gx failed!\n");return;}

    cudaStatus = cudaMallocArray(&d_xrayGy1, &channelDescImage,W,H);
    if(cudaStatus!=cudaSuccess){printf("Allocating the x-ray device image Gy failed!\n");return;}

}

void PatchRayCast::AllocateCT()
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


void PatchRayCast::ComputeXrayGradient(int xraynum)
{

    int i=0;
    int j=0;
    int W=(int)h_camera1[0];
    int H=(int)h_camera1[1];

    if(xraynum==1)
    {
        i=0;
        for(j=0; j<H; j++)
        {
            h_xrayGx1[H*i+j] = 0;
        }
        for(i=1; i<(W-1); i++)
        {
            for(j=0; j<H; j++)
            {
                h_xrayGx1[H*i+j] = 0.5*( h_xray1[H*(i-1)+j] - h_xray1[H*(i+1)+j] );
            }
        }
        i=(W-1);
        for(j=0; j<H; j++)
        {
            h_xrayGx1[H*i+j] = 0;
        }

        j=0;
        for(i=0; i<W; i++)
        {
            h_xrayGy1[H*i+j] = 0;
        }
        for(j=1; j<(H-1); j++)
        {
            for(i=0; i<W; i++)
            {
                h_xrayGy1[H*i+j] = 0.5*(h_xray1[H*i+(j-1)] - h_xray1[H*i+(j+1)]);
            }
        }
        j=(H-1);
        for(i=0; i<H; i++)
        {
            h_xrayGy1[H*i+j] = 0;
        }

    }
    else
    {
        i=0;
        for(j=0; j<H; j++)
        {
            h_xrayGx2[H*i+j] = 0;
        }
        for(i=1; i<(W-1); i++)
        {
            for(j=0; j<H; j++)
            {
                h_xrayGx2[H*i+j] = 0.5*(h_xray2[H*(i-1)+j] - h_xray2[H*(i+1)+j]);
            }
        }
        i=(W-1);
        for(j=0; j<H; j++)
        {
            h_xrayGx2[H*i+j] = 0;
        }

        j=0;
        for(i=0; i<W; i++)
        {
            h_xrayGy2[H*i+j] = 0;
        }
        for(j=1; j<(H-1); j++)
        {
            for(i=0; i<W; i++)
            {
                h_xrayGy2[H*i+j] = 0.5*(h_xray2[H*i+(j-1)] - h_xray2[H*i+(j+1)]);
            }
        }
        j=(H-1);
        for(i=0; i<H; i++)
        {
            h_xrayGy2[H*i+j] = 0;
        }

    }

}



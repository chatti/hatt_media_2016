#ifndef RAYCAST_H
#define RAYCAST_H

#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>

class RayCast
{

    int numXray;

    float ngcVal;
    float nccVal;

	//The input X-ray image and gradients
    float* h_xray1;
    float* h_xrayGx1;
    float* h_xrayGy1;
    float* d_xray1;
    float* d_xrayGx1;
    float* d_xrayGy1;

    float* h_xray2;
    float* h_xrayGx2;
    float* h_xrayGy2;
    float* d_xray2;
    float* d_xrayGx2;
    float* d_xrayGy2;

	//The DRRs and gradients
    float* h_DRR1;
    float* h_DRRGx1;
    float* h_DRRGy1;
    float* d_DRR1;
    float* d_DRRGx1;
    float* d_DRRGy1;

    //The DRRs and gradients
    float* h_DRR2;
    float* h_DRRGx2;
    float* h_DRRGy2;
    float* d_DRR2;
    float* d_DRRGx2;
    float* d_DRRGy2;

    //The CT image
    float* h_CT;
    cudaArray* d_CT;

	//Variables used for NCC calculation
    float* d_vector1;
    float* d_vector2;
    float* d_mean;
    float* d_std;
    float* d_cost;

    float  h_xraymean1;
    float  h_xraymean2;
    float  h_xraystd1;
    float  h_xraystd2;
    float  h_xrayGxmean1;
    float  h_xrayGxmean2;
    float  h_xrayGxstd1;
    float  h_xrayGxstd2;
    float  h_xrayGymean1;
    float  h_xrayGymean2;
    float  h_xrayGystd1;
    float  h_xrayGystd2;

    float  h_cost[1];
    float  h_std[1];

	//Variables used for the 3D texture
	cudaChannelFormatDesc channelDescCT;
	cudaExtent volSize;
	//These are camera parameters that determine:
	//detector width in pixels   = camera[0]
	//detector height in pixels  = camera[1]
	//detector element spacing x = camera[2]
	//detector element spacing y = camera[3]
	//detector sid (f)           = camera[4]
    float h_camera1[12];
    float h_camera2[12];
	//These are CT parameters that determine:
    //Width X                  h_ctParams[0]
    //Width Y	           h_ctParams[1]
    //Width Z	           h_ctParams[2]
    //PixelDimensions          h_ctParams[3]
    //PixelDimensions          h_ctParams[4]
    //PixelDimensions          h_ctParams[5]
    //CenterPixel              h_ctParams[6]
    //CenterPixel         	   h_ctParams[7]
    //CenterPixel	           h_ctParams[8]
    float h_ctParams[9];
	//These are Ray parameters that determine:
	//z path length			      		rayParams[0]
	//total ray steps			      	rayParams[1]
    float h_rayParams1[4];
    float h_rayParams2[4];
	/////////////////////////////////////////


    //Parameters for spatial transformations (host and device)
    float  h_initial[6];        //The initial spatial transform
    float  h_secondary[6];      //The second  spatial transform
    float  h_CT2Cam1[16];		//The full spatial transform, including the camera (primary)
    float  h_CT2Cam2[16];		//The full spatial transform, including the camera (primary)
    float  h_Cam2CT1[16];		//The full spatial transform, including the camera (primary)
    float  h_Cam2CT2[16];		//The full spatial transform, including the camera (primary)
    float  h_iT[16];
    float  h_iR[16];
    float  h_iTinv[16];
    float  h_iRinv[16];
    float  h_P2W[16];          //P2W = T*iT*R*iR
    float  h_W2P[16];          //W2P = P2W^-1
    float  h_CT2P[16];         // CT2P = OS = center of rotation * spatial resolution
    float  h_P2CT[16];         // P2CT = CT2P^-1
    float  h_W2Cam1[16];       // W2Cam = World to Camera coordinate system
    float  h_Cam2W1[16];       // Cam2W = W2Cam^-1
    float  h_W2Cam2[16];       // W2Cam = World to Camera coordinate system
    float  h_Cam2W2[16];       // Cam2W = W2Cam^-1

    float  TMP1[16];		//Dummy matrices
    float  TMP2[16];		//Dummy matrices


    //This is the CT bounding box used to create a bounding box projection

    float h_BoxCT[32];
    int h_BoundingBox1[4];
    int h_BoundingBox2[4];

	//For tracking the status of the GPU
	cudaError_t cudaStatus;
	//General status integer 0:ok, -1:Read issue, -2:Write issue, -3:
	int cpuStatus;

public:
    RayCast();

    ~RayCast();
	void StartUpGPU();

    void SetRayParams1(float start, float stop, float dx, float dy, float dz);
    void SetRayParams2(float start, float stop, float dx, float dy, float dz);

    void SetTransformParams1(float tx, float ty, float tz, float rx, float ry, float rz);
    void SetTransformParams2(float tx, float ty, float tz, float rx, float ry, float rz);
    void SetCTParams(float xdim, float ydim, float zdim, float dx, float dy, float dz, float cx, float cy, float cz);

    void ReadXray1(std::string  filename, float W, float H);
    void ReadXray2(std::string  filename, float W, float H);
    void ReadCT(std::string filename);

    void TransferXrayGPU1();
    void TransferXrayGPU2();
	void TransferCTGPU();

    void CreateRayCastGPU();
    void TransferConstantsToGPU();
	void WriteImage(char* filename, float* img, int W, int H);
	void WriteVolume(char* filename, float* vol, int sx, int sy, int sz);
    void WriteDRR(char* filename, int num);
    void WriteDRRGx(char* filename, int num);
    void WriteDRRGy(char* filename, int num);
    void WriteXrayGx(char* filename, int num);
    void WriteXrayGy(char* filename, int num);
    void ReadImage(std::string filename, float* img, int W, int H);
    void ReadVolume(std::string filename, float* vol, int sx, int sy, int sz);

    void AllocateXray1();
    void AllocateXray2();
    void AllocateCT();

    void TransferXrayToHost1();
    void TransferXrayGradientToHost1();
    void TransferXrayToHost2();
    void TransferXrayGradientToHost2();

    void TransferDRRToHost1();
    void TransferDRRGradientToHost1();
    void TransferDRRToHost2();
    void TransferDRRGradientToHost2();

	bool isOK();

    void MM44(float (&MO)[16], float (&ML)[16], float (&MR)[16]);
    void MM48(float (&MO)[32], float (&ML)[16], float (&MR)[32]);

    void PrintM44(float (&M)[16]);
    void PrintM48(float (&M)[32]);
    void SetEye(float (&M)[16]);
    void SetSpatialTranslation(float (&M)[16],float* tform);
    void SetSpatialTranslationInverse(float (&M)[16],float* tform);
    void SetSpatialRotation(float (&M)[16],float* tform);
    void SetSpatialRotationInverse(float (&M)[16],float* tform);

    void InitForwardMatrices();
    void InitInverseMatrices();
    void UpdateTransformMatrix();

    void SetInitialTransform(float tx, float ty, float tz, float rx, float ry, float rz);
    void SetSecondaryTransform(float tx, float ty, float tz, float rx, float ry, float rz);

    void ReadParameterFile(std::string filename);

    void ComputeBoundingBox1();
    void ComputeBoundingBox2();
    void PrintBoundingBox(int num);

    void CreateCTBoundingBox();

    float ComputeImageMean(float* img, int W, int H);
    float ComputeImageStd(float* img, float mean, int W, int H);

    int GetNumXray();

    float ComputeNCC();
    float ComputeNGC();

    void ComputeDRRGradient1();
    void ComputeDRRGradient2();

    void PrintNGC();
    void PrintNCC();

};

#endif

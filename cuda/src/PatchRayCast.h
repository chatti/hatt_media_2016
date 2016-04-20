#ifndef PATCHRAYCAST_H
#define PATCHRAYCAST_H

#define PATCHCOUNT 18

#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>

class PatchRayCast
{

    int numXray;

    float ngcVal;

	//The input X-ray image and gradients
    float* h_xray1;
    float* h_xrayGx1;
    float* h_xrayGy1;
    cudaArray* d_xrayGx1;
    cudaArray* d_xrayGy1;

    float* h_xray2;
    float* h_xrayGx2;
    float* h_xrayGy2;
    cudaArray* d_xrayGx2;
    cudaArray* d_xrayGy2;

    //The CT image
    float* h_CT;
    cudaArray* d_CT;

    float  h_PGCC[2*PATCHCOUNT];
    float* d_PGCC;
    float PGCC;

    //2D texture
    cudaChannelFormatDesc channelDescImage;
	//Variables used for the 3D texture
	cudaChannelFormatDesc channelDescCT;
	cudaExtent volSize;
	//These are camera parameters that determine:

    float h_camera1[12];
    float h_camera2[12];
	//These are CT parameters that determine:

    float h_ctParams[9];
	//These are Ray parameters that determine:
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
    float  h_P1[16];
    float  h_P2[16];

    float  TMP1[16];		//Dummy matrices
    float  TMP2[16];		//Dummy matrices
    float  A[4*PATCHCOUNT];		//Dummy matrices
    float  B[4*PATCHCOUNT];		//Dummy matrices


    //This is the CT bounding box used to create a bounding box projection

    float h_KeyPointsCT[4*PATCHCOUNT];
    float h_PatchLocations1[2*PATCHCOUNT];
    float h_PatchLocations2[2*PATCHCOUNT];

	//For tracking the status of the GPU
	cudaError_t cudaStatus;
	//General status integer 0:ok, -1:Read issue, -2:Write issue, -3:
	int cpuStatus;

public:
    PatchRayCast();
    ~PatchRayCast();
	void StartUpGPU();

    void SetRayParams1(float start, float stop, float dx, float dy, float dz);
    void SetRayParams2(float start, float stop, float dx, float dy, float dz);

    void SetTransformParams1(float tx, float ty, float tz, float rx, float ry, float rz);
    void SetTransformParams2(float tx, float ty, float tz, float rx, float ry, float rz);
    void SetCTParams(float xdim, float ydim, float zdim, float dx, float dy, float dz, float cx, float cy, float cz);

    void ReadXray1(std::string  filename, float W, float H);
    void ReadXray2(std::string  filename, float W, float H);
    void ReadCT(std::string filename);

    void TransferXrayGPU(int xraynum);
	void TransferCTGPU();

    void TransferConstantsToGPU();
    void WriteImage(std::string filename, float* img, int W, int H);
    void WriteVolume(std::string filename, float* vol, int sx, int sy, int sz);
    void WriteDRR(std::string filename, int num);
    void WriteDRRGx(std::string filename, int num);
    void WriteDRRGy(std::string filename, int num);
    void WriteXrayGx(std::string filename, int num);
    void WriteXrayGy(std::string filename, int num);
    void ReadImage(std::string filename, float* img, int W, int H);
    void ReadVolume(std::string filename, float* vol, int sx, int sy, int sz);

    void AllocateXray1();
    void AllocateXray2();
    void AllocateCT();

    void ComputeXrayGradient(int xraynum);

	bool isOK();

    void MM4_4(float (&O)[16], float (&L)[16], float (&R)[16]);
    void MM4_PC(float (&O)[4*PATCHCOUNT], float (&L)[16], float (&R)[4*PATCHCOUNT]);

    void PrintM44(float (&M)[16]);
    void PrintM4PC(float (&M)[4*PATCHCOUNT]);
    void SetEye(float (&M)[16]);
    void SetSpatialTranslation(float (&M)[16],float* tform);
    void SetSpatialTranslationInverse(float (&M)[16],float* tform);
    void SetSpatialRotation(float (&M)[16],float* tform);
    void SetSpatialRotationInverse(float (&M)[16],float* tform);

    void InitForwardMatrices();
    void InitInverseMatrices();
    void UpdateTransformMatrix();

    void Hom2Screen1(float (&M)[4*PATCHCOUNT]);
    void Hom2Screen2(float (&M)[4*PATCHCOUNT]);

    void SetInitialTransform(float tx, float ty, float tz, float rx, float ry, float rz);
    void SetSecondaryTransform(float tx, float ty, float tz, float rx, float ry, float rz);

    void ReadParameterFile(std::string filename);
    void UpdatePatchLocations1();
    void UpdatePatchLocations2();

    int GetNumXray();
    float ComputePatchGCC();


};

#endif

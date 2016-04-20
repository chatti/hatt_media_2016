#ifndef DASH_H
#define DASH_H

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <string>
#include <fstream>
#include <cuda_runtime.h>


class DASH
{
    int numXray;
	int T;
	int B;
    int metric;
	//The input X-ray image and gradients
	float* h_xray1;
    float* h_xrayGx1;
    float* h_xrayGy1;
	cudaArray* d_xray1;
	cudaArray* d_xrayGx1;
	cudaArray* d_xrayGy1;

	float* h_xray2;
    float* h_xrayGx2;
    float* h_xrayGy2;
	cudaArray* d_xray2;
	cudaArray* d_xrayGx2;
	cudaArray* d_xrayGy2;

	//The point cloud
	float* h_pcloud;
	float* d_pcloud;
	//The point cloud
	float* h_gcloud;
	float* d_gcloud;

	//These are camera parameters that determine:
	//detector width in pixels   = camera[0]
	//detector height in pixels  = camera[1]
	//detector element spacing x = camera[2]
	//detector element spacing y = camera[3]
	//detector sid (f)           = camera[4]
	//detector isocenter         = camera[5]
	//spatial transform          = camera[6-11]
	float h_camera1[12];
	float h_camera2[12];

	//The spatial transform parameters of the camera
	//These are point cloud parameters that determine:
	//N - number of points         pcParams[0]
	
	cudaChannelFormatDesc channelDescXray;
	float pcloudSize;
	/////////////////////////////////////////

	//Cost Function Values for host and device
	float* h_cost;
	float* d_cost;
    float  h_finalCost[1];

	//Parameters for spatial transformations (host and device)
	float  h_init[6];       //The initial spatial transform
	float  h_transform[6];  //The second  spatial transform
	float  h_M1[16];		//The full spatial transform, including the camera (primary)
	float  h_M2[16];        //The full spatial transform, including the camera (second)
    float  h_F[16];             //F = T*iT*R*iR
    float  h_T[16];		    //2nd transform translation
	float  h_R[16];		    //2nd transform rotation
	float  h_iT[16];		//Init translation
	float  h_iR[16];		//Init rotation
    float  h_C1[16];		//Camera 1
    float  h_C2[16];		//Camera 2
    float  TMP1[16];		    //Dummy matrices
    float  TMP2[16];		    //Dummy matrices
	float* d_M1;
	float* d_M2;

	
	//For tracking the status of the GPU
	cudaError_t cudaStatus;
	//General status integer 0:ok, -1:Read issue, -2:Write issue, -3:
	int cpuStatus;

public:
	DASH();
	~DASH();


	void StartUpGPU();
    int GetNumXray();
    void SetNumXray(int num);
    void SetMetric(int m);
	void SetCamera1Params(float f, float W, float H, float sx, float sy, float iso, float* phi);
	void SetCamera2Params(float f, float W, float H, float sx, float sy, float iso, float* phi);
    void SetInitialTransform(float tx, float ty, float tz, float rx, float ry, float rz);
    void SetTransform(float tx, float ty, float tz, float rx, float ry, float rz);
	void SetPCloudParams(float N);
    void ReadXray1(std::string filename);
    void ReadXray2(std::string filename);
    void ReadXrayGx1(std::string filename);
    void ReadXrayGx2(std::string filename);
    void ReadXrayGy1(std::string filename);
    void ReadXrayGy2(std::string filename);
    void ReadPCloud(std::string filename);
    void ReadGCloud(std::string filename);
	void TransferXray1ToGPU();
	void TransferXray2ToGPU();
	void TransferPCloudToGPU();
	void TransferGCloudToGPU();
	void TransferConstantsToGPU();
    float ComputeDASHCostFunction();
    void WriteImage(std::string filename, float* img, int W, int H);
    void WriteVolume(std::string filename, float* vol, int sx, int sy, int sz);
    void ReadImage(std::string filename, float* img, int W, int H);
    void ReadParameterFile(std::string filename);
	bool isOK();
    void MM44(float (&MO)[16], float (&ML)[16], float (&MR)[16]);
    void PrintM44(float (&M)[16]);
    void SetEye(float (&M)[16]);
    void SetSpatialTranslation(float (&M)[16],float* tform);
    void SetSpatialRotation(float (&M)[16],float* tform);
    void InitAllMatrices();
    void UpdateTransformMatrix();

};

#endif

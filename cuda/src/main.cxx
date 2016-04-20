#include "RayCastDRR.cxx"
#include "SplatDRR.cxx"
#include "DASH.cxx"

#include <stdio.h>
#include <cuda_runtime.h>

int RunRayCast();
int RunSplat();
int RunDASH();

int main(){
	return RunRayCast();
	//return RunSplat();
	//return RunDASH();
}

int RunRayCast(){

RayCastDRR X(1000.0, .15, .15, 512, 512, 
			     432, 150, 146,
			     0.1037, 0.1037, 0.1037,
			     12.23, 11.82, 7.77);
	
	if( !X.isOK() )
	{
		printf("Something went wrong with the initialization!\n");
		return 1;
	}

	//X.ReadXray((char*)"DRR.img");
	X.ReadCT((char*)"ct.raw");
	X.SetTransformParams(0,0,300,0,0,0);
	X.SetRayParams(25.0,64);
	X.TransferConstantParamsGPU();

	float ncc=0.0;

	///////////Timing GPU DRR//////////////////
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	//X.PerformTextureTest();
	for( int i=0; i< 100; i++){
	X.CreateRayCastDRRGPU();
	X.ComputeGradient();
	ncc = X.ComputeNCC();
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	///////////////////////////////////////////

	X.TransferDRRToHost();
	X.TransferDRRGradientToHost();
	X.WriteDRR((char*)"DRR.raw");
	X.WriteDRRGx((char*)"DRRGx.raw");
	X.WriteDRRGy((char*)"DRRGy.raw");

	printf("Time (ms): %f\n",milliseconds/100);
	return 0;
}

int RunSplat()
{
	//SID =1000.0 mm, image 512 x 512 pixels, detector element spacing (mm/px), and point cloud size
	int N = (int)pow(2,18);
	SplatDRR X(1000.0, 512, 512, 0.15, 0.15,N );
	
	if( !X.isOK() ){printf("Something went wrong with the initialization!\n");return 1;}

	X.ReadXray((char*)"xray.raw");
	X.ReadPC((char*)"pc.pnt");
	X.SetTransformParams(0,0,300,0,0,0);
	X.TransferConstantParamsToGPU();

	float ncc=0.0;

	///////////Timing GPU DRR//////////////////
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	for( int i=0; i< 100; i++){
	X.CreateSplatDRRGPU();
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	///////////////////////////////////////////

	X.WriteDRR((char*)"DRR.img");
	X.WritePC2D((char*)"PC2D.pnt");

	printf("Time (ms): %f\n",milliseconds/100);
	return 0;
}

int RunDASH()
{
	//SID =1000.0 mm, image 512 x 512 pixels, detector element spacing (mm/px), and point cloud size
	int N = (int)pow(2,18);
	DASH X(1000.0, 512, 512, 0.15, 0.15,N );
	
	if( !X.isOK() ){printf("Something went wrong with the initialization!\n");return 1;}

	X.ReadXray((char*)"xray.raw");
	X.ReadPC((char*)"pc.pnt");
	X.SetTransformParams(0,0,300,0,0,0);
	X.TransferConstantParamsToGPU();

	float cc = 0.0;

	///////////Timing GPU DRR//////////////////
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	for( int i=0; i< 100; i++){
		cc = X.ComputeDASHGPU();
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	///////////////////////////////////////////

	printf("Time (ms): %f\n",milliseconds/100);
	printf("CC Value : %f\n",cc);
	return 0;
}

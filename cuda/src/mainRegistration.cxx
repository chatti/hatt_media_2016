#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include "RayCastDRR.h"
#include "DASH.h"
#include "RayCastDRRCostFunction.h"
#include <vnl/algo/vnl_lbfgs.h>  //limited memory BFGS algorithm (general unconstrained optimization)
#include <vnl/algo/vnl_lbfgsb.h> //constrained BFGS
#include <vnl/algo/vnl_amoeba.h> //Nelder-mead
#include <vnl/algo/vnl_powell.h> //Powell
vector<vector<float> > Amoeba(RayCastCostFunction &cf, vnl_vector<double> iTransform);
vector<vector<float> > Powell(RayCastCostFunction &cf, vnl_vector<double> iTransform);

int main(int argc, char* args[] )
{

    //argc should be at least 1+1+5+9+2+6=24
    //(exename + ctfilename + camera + ct + ray + phi)
    int expectedNumberOfArguments=25;
    if( argc!=expectedNumberOfArguments)
    {

        std::cout << "This program should have 24 input arguments, not " << argc << "\n";

        std::cout << "CT filename: string\n";
        std::cout << "X-ray filename: string\n";
        std::cout << "C-arm SID: float\n";
        std::cout << "C-arm image width (#px): float\n";
        std::cout << "C-arm image height (#px): float\n";
        std::cout << "C-arm detector x spacing (mm/px): float\n";
        std::cout << "C-arm detector y spacing (mm/px): float\n";
        std::cout << "CT # voxels in x-dim: int\n";
        std::cout << "CT # voxels in y-dim: int\n";
        std::cout << "CT # voxels in z-dim: int\n";
        std::cout << "CT voxel spacing in x-dim (mm/vx): float\n";
        std::cout << "CT voxel spacing in x-dim (mm/vx): float\n";
        std::cout << "CT voxel spacing in x-dim (mm/vx): float\n";
        std::cout << "CT center x-dim (mm): float\n";
        std::cout << "CT center y-dim (mm): float\n";
        std::cout << "CT center z-dim (mm): float\n";
        std::cout << "Ray cast z-range x-dim (mm): float\n";
        std::cout << "Ray cast number of steps (controls granularity): int\n";
        std::cout << "Initial transform tx (mm): float\n";
        std::cout << "Initial transform ty (mm): float\n";
        std::cout << "Initial transform tz (mm): float\n";
        std::cout << "Initial transform rx (radians): float\n";
        std::cout << "Initial transform ry (radians): float\n";
        std::cout << "Initial transform rz (radians): float\n";
        return 1;
    }

    std::string ctFileName="";
    std::string xrayFileName="";

    float params[expectedNumberOfArguments-3];
    std::stringstream ss;
    ss << args[1]; ss >> ctFileName; ss.str(""); ss.clear();
    ss << args[2]; ss >> xrayFileName; ss.str(""); ss.clear();

    for( int k=3; k < expectedNumberOfArguments; k++){
        ss << args[k]; ss >> params[k-3]; ss.str(""); ss.clear();
    }

    float sid = params[0];
    float W = params[1];
    float H = params[2];
    float pitchX = params[3];
    float pitchY = params[4];
    float ctSizeX = params[5];
    float ctSizeY = params[6];
    float ctSizeZ = params[7];
    float ctResX = params[8];
    float ctResY = params[9];
    float ctResZ = params[10];
    float ctCenterX = params[11];
    float ctCenterY = params[12];
    float ctCenterZ = params[13];
    float rayzrange = params[14];
    float raySteps = params[15];
    float tx = params[16];
    float ty = params[17];
    float tz = params[18];
    float rx = params[19];
    float ry = params[20];
    float rz = params[21];

    std::cout << "CT filename: " << ctFileName << std::endl;
    std::cout << "Xray filename: " << xrayFileName << std::endl;
    std::cout << "C-arm SID: " << sid << std::endl;
    std::cout << "C-arm image width (#px): " << W << std::endl;
    std::cout << "C-arm image height (#px): " << H << std::endl;
    std::cout << "C-arm detector x spacing (mm/px): " << pitchX << std::endl;
    std::cout << "C-arm detector y spacing (mm/px): " << pitchY << std::endl;
    std::cout << "CT # voxels in x-dim: " << ctSizeX << std::endl;
    std::cout << "CT # voxels in y-dim: " << ctSizeY << std::endl;
    std::cout << "CT # voxels in z-dim: " << ctSizeZ << std::endl;
    std::cout << "CT voxel spacing in x-dim (mm/vx): " << ctResX << std::endl;
    std::cout << "CT voxel spacing in x-dim (mm/vx): " << ctResY << std::endl;
    std::cout << "CT voxel spacing in x-dim (mm/vx): " << ctResZ << std::endl;
    std::cout << "CT center x-dim (mm): " << ctCenterX << std::endl;
    std::cout << "CT center y-dim (mm): " << ctCenterY << std::endl;
    std::cout << "CT center z-dim (mm): " << ctCenterZ << std::endl;
    std::cout << "Ray cast path length in z (mm): " << rayzrange << std::endl;
    std::cout << "Ray cast number of steps (controls granularity): " << raySteps << std::endl;
    std::cout << "Probe transform tx: " << tx << std::endl;
    std::cout << "Probe transform ty: " << ty << std::endl;
    std::cout << "Probe transform tz: " << tz << std::endl;
    std::cout << "Probe transform rx: " << rx << std::endl;
    std::cout << "Probe transform ry: " << ry << std::endl;
    std::cout << "Probe transform rz: " << rz << std::endl;

    float ctParams[9];
    float xrayParams[5];
    for(int k=0; k<9;k++)
    {
        ctParams[k]=params[k+5];
    }
    vnl_vector<double> iTransform(6);
    for(int k=0; k<6;k++)
    {
        iTransform[k]=(double)params[k+16];
    }

    xrayParams[0] = params[1];
    xrayParams[1] = params[2];
    xrayParams[2] = params[3];
    xrayParams[3] = params[4];
    xrayParams[4] = params[0];

    RayCastCostFunction cf(6);
    cf.Initialize((char*)xrayFileName.c_str(), xrayParams, (char*)ctFileName.c_str(), ctParams);
    printf("Cost Function Initialized\n");

    vector<vector<float> > optimizationLog = Amoeba(cf,iTransform);
    //vector<vector<float> > optimizationLog = Powell(cf,iTransform);

    for( int k=0; k < optimizationLog.size();k++)
    {
        vector<float> tmp = optimizationLog[k];
        printf("%d: ",k);
        for( int j=0; j < tmp.size();j++)
        {
            printf("%.2f ", tmp[j]);

        }
        printf("\n");
    }
    printf("Registration Finished\n");
    return 0;
}


vector<vector<float> > Amoeba(RayCastCostFunction &cf, vnl_vector<double> iTransform)
{
    vnl_amoeba Minimizer(cf);

    Minimizer.set_f_tolerance(0.0001);
    Minimizer.set_x_tolerance(0.0001);
    Minimizer.set_max_iterations (500);

	//you can define the scale of the space ???
    vnl_vector<double> dx(6);
	dx(0) = 1.0;dx(1) = 1.0;dx(2) = 1.0;
	dx(3) = 1.0;dx(4) = 1.0;dx(5) = 1.0;

    cout << "Minimizing using Amoeba" << endl;
    cout << "Started at: " << iTransform << endl;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	///////////////////////////

    Minimizer.minimize(iTransform);
	
	//////////////////////////
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	
    cout << "Ended at: " << iTransform << endl;
	cout << "NumEvals: " << Minimizer.get_num_evaluations() << endl;
	cout << "Time (ms): " << milliseconds << endl;
    return cf.GetHistory();
}

vector<vector<float> > Powell(RayCastCostFunction &cf, vnl_vector<double> iTransform)
{
    vnl_powell Minimizer(&cf);

    Minimizer.set_f_tolerance(0.001);
    Minimizer.set_x_tolerance(0.001);
    //Minimizer.set_max_iterations (500);

    //you can define the scale of the space ???
    vnl_vector<double> dx(6);
    dx(0) = 1.0;dx(1) = 1.0;dx(2) = 1.0;
    dx(3) = 1.0;dx(4) = 1.0;dx(5) = 1.0;

    cout << "Minimizing using Amoeba" << endl;
    cout << "Started at: " << iTransform << endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    ///////////////////////////

    Minimizer.minimize(iTransform);

    //////////////////////////
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Ended at: " << iTransform << endl;
    cout << "NumEvals: " << Minimizer.get_num_evaluations() << endl;
    cout << "Time (ms): " << milliseconds << endl;
    return cf.GetHistory();
}

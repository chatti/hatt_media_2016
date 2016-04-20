#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include "RayCast.h"
#include "RayCastCostFunction.h"

int main(int argc, char* args[] ){

    if(argc != 4)
    {
        printf("Proper usage: raycastDRR [parameterfile] [outputfilename] [ComputeGradients 0=no, 1=yes] \n");
        return -1;
    }

    std::string pfile;
    std::string outputfile;
    int computeGradients;

    std::stringstream ss;
    ss << args[1]; ss >> pfile;   ss.str(""); ss.clear();
    ss << args[2]; ss >> outputfile; ss.str(""); ss.clear();
    ss << args[3]; ss >> computeGradients; ss.str("");ss.clear();


    RayCast X;
    X.ReadParameterFile(pfile);
    if(!X.isOK()){
        return -1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    ///////////////////////////

    X.CreateRayCastGPU();

    if(computeGradients==1)
    {
        X.ComputeDRRGradient1();
    }

    //////////////////////////
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time (ms): " << milliseconds << endl;

    //X.PrintBoundingBox(1);
    X.TransferDRRToHost1();        
    X.WriteDRR((char*)outputfile.c_str(),1);

    if(computeGradients)
    {
        X.TransferDRRGradientToHost1();
        X.WriteDRRGx((char*)"DRR_Gx.raw",1);
        X.WriteDRRGy((char*)"DRR_Gy.raw",1);
        X.WriteXrayGx((char*)"simxrayGx.raw",1);
        X.WriteXrayGy((char*)"simxrayGy.raw",1);

    }

    printf("DRR Finished\n");
    return 0;
}




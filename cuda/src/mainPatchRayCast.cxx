#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include "PatchRayCast.h"
#include "PatchRayCastCostFunction.h"
#include <vnl/algo/vnl_lbfgs.h>  //limited memory BFGS algorithm (general unconstrained optimization)
#include <vnl/algo/vnl_lbfgsb.h> //constrained BFGS
#include <vnl/algo/vnl_amoeba.h> //Nelder-mead
#include <vnl/algo/vnl_powell.h> //Powell
FloatMatrixType Amoeba(PatchRayCastCostFunction &cf,int iter);
FloatMatrixType Powell(PatchRayCastCostFunction &cf, double stepsize);
FloatMatrixType LBFGS(PatchRayCastCostFunction &cf);
FloatMatrixType Biplane(PatchRayCast &X, int method, int metric, std::vector<float> stepsizes);
FloatMatrixType Monoplane(PatchRayCast &X, int method, int metric, std::vector<float> stepsizes);

void WriteHistory(std::string logFileName, FloatMatrixType optimizationLog);

int main(int argc, char* args[] ){

    if(argc < 5)
    {
        printf("Proper usage: patchraycast [parameterfile] [logfile] [OptAlgorithm (1=Amoeba 2=Powell 3=LFBGS)] [Similarity Metric (1=NCC 2= Gradient NCC)] ]\n");
        return -1;
    }

    std::string pfile;
    std::string logfile;
    std::string resfile;
    std::vector<float> stepsizes(3,0);
    int method = 0;
    int metric = 0;

    std::stringstream ss;
    ss << args[1]; ss >> pfile;   ss.str(""); ss.clear();
    ss << args[2]; ss >> logfile; ss.str(""); ss.clear();
    ss << args[3]; ss >> method;  ss.str(""); ss.clear();
    ss << args[4]; ss >> metric;  ss.str(""); ss.clear();
    metric = 2;
    //ss << args[5]; ss >> stepsizes[0];  ss.str(""); ss.clear();
    //ss << args[6]; ss >> stepsizes[1];  ss.str(""); ss.clear();
    //ss << args[7]; ss >> stepsizes[2];  ss.str(""); ss.clear();

    stepsizes[0]=1;
    stepsizes[1]=.1;
    stepsizes[2]=.01;


    PatchRayCast X;
    X.ReadParameterFile(pfile);
    if(!X.isOK()){
        return -1;
    }

    FloatMatrixType allLog;
    if(X.GetNumXray() == 1)
    {
        allLog = Monoplane(X,method,metric,stepsizes);
    }
    else
    {
        allLog = Biplane(X,method,metric,stepsizes);
    }

    WriteHistory(logfile, allLog);
    printf("Registration Finished\n");
    return 0;
}


FloatMatrixType Monoplane(PatchRayCast &X, int method, int metric, std::vector<float> stepsizes)
{

    bool usegradient = false;
    if(metric == 1)
    {
        usegradient = false;
    }
    else
    {
        usegradient = true;
    }

    PatchRayCastCostFunction cf12___6(3);
    cf12___6.SetPatchRayCast(&X);
    PatchRayCastCostFunction cf12_456(5);
    cf12_456.SetPatchRayCast(&X);
    PatchRayCastCostFunction cf123456(6);
    cf123456.SetPatchRayCast(&X);

    printf("Cost Function Initialized\n");

    vnl_vector_fixed<bool,6> idx12___6;
    idx12___6(0)=true;
    idx12___6(1)=true;
    idx12___6(2)=false;
    idx12___6(3)=false;
    idx12___6(4)=false;
    idx12___6(5)=true;
    cf12___6.SetIdx(idx12___6);

    vnl_vector_fixed<bool,6> idx12_456;
    idx12_456(0)=true;
    idx12_456(1)=true;
    idx12_456(2)=false;
    idx12_456(3)=true;
    idx12_456(4)=true;
    idx12_456(5)=true;
    cf12_456.SetIdx(idx12_456);

    vnl_vector_fixed<bool,6> idx123456;
    idx123456(0)=true;
    idx123456(1)=true;
    idx123456(2)=true;
    idx123456(3)=true;
    idx123456(4)=true;
    idx123456(5)=true;
    cf123456.SetIdx(idx123456);

    vnl_vector_fixed<float,6> init;
    init(0)=0.0;
    init(1)=0.0;
    init(2)=0.0;
    init(3)=0.0;
    init(4)=0.0;
    init(5)=0.0;
    cf12___6.SetInit(init);
    cf12_456.SetInit(init);
    cf123456.SetInit(init);

    vnl_vector_fixed<float,6> offsetL;
    vnl_vector_fixed<float,6> offsetS;

    offsetL(0)=5.0;
    offsetL(1)=5.0;
    offsetL(2)=5.0;
    offsetL(3)=3.14/4;
    offsetL(4)=3.14/4;
    offsetL(5)=3.14/4;
    offsetS(0)=0.5;
    offsetS(1)=0.5;
    offsetS(2)=0.5;
    offsetS(3)=3.14/16;
    offsetS(4)=3.14/16;
    offsetS(5)=3.14/16;

    cf12___6.SetOffset(offsetL);
    cf12_456.SetOffset(offsetL);
    cf123456.SetOffset(offsetS);

    std::vector<FloatMatrixType> logs;
    FloatMatrixType log;

    //cout << "Minimizing using Amoeba" << endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    ///////////////////////////


    if(method == 1)
    {
        log = Amoeba(cf12___6, 500); logs.push_back(log);
        cf12_456.SetInit(cf12___6.GetTransform());
        log = Amoeba(cf12_456, 1500); logs.push_back(log);
        //cf123456.SetInit(cf12_456.GetTransform());
        //log = Amoeba(cf123456, 1000); logs.push_back(log);
    }
    else if( method == 2)
    {

        log = Powell(cf12___6,2.5); logs.push_back(log);
        cf12_456.SetInit(cf12___6.GetTransform());
        log = Powell(cf12_456,0.5); logs.push_back(log);
        //cf123456.SetInit(cf12_456.GetTransform());
        //log = Powell(cf123456,.005); logs.push_back(log);
        //cf123456.f((const vnl_vector<double>)cf123456.ComputeStartingVector());
    }
    else if( method ==3)
    {
        log = LBFGS(cf12___6); logs.push_back(log);
        cf12_456.SetInit(cf12___6.GetTransform());
        log = LBFGS(cf12_456); logs.push_back(log);
        //cf123456.SetInit(cf12_456.GetTransform());
        //log = LBFGS(cf123456); logs.push_back(log);
    }
    //////////////////////////
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time (ms): " << milliseconds << std::endl;
    std::cout << "Final Transform: ";
    for(int k=0; k<6; k++)
    {
        std::cout << cf123456.GetTransform()[k] << " ";
    }
    std::cout << std::endl;

    FloatMatrixType allLog;
    for( int k=0; k < logs.size(); k++)
    {
        log = logs[k];
        for( int m=0; m < log.size(); m++)
        {
            allLog.push_back(log[m]);
        }
    }


    std::vector<float> finalTransform;
    vnl_vector_fixed<float,6> ft = cf123456.GetTransform();
    for( int k=0; k<6; k++ )
    {
        finalTransform.push_back(ft[k]);
    }

    allLog.push_back(finalTransform);

    std::vector<float> timing;
    timing.push_back(milliseconds);
    allLog.push_back(timing);

    return allLog;

}

FloatMatrixType Biplane(PatchRayCast &X, int method, int metric, std::vector<float> stepsizes)
{
    bool usegradient = false;
    if(metric == 1)
    {
        usegradient = false;
    }
    else
    {
        usegradient = true;
    }
    PatchRayCastCostFunction cf12___6(3);
    cf12___6.SetPatchRayCast(&X);
    PatchRayCastCostFunction cf12_456(5);
    cf12_456.SetPatchRayCast(&X);
    PatchRayCastCostFunction cf123456(6);
    cf123456.SetPatchRayCast(&X);

    printf("Cost Function Initialized\n");

    vnl_vector_fixed<bool,6> idx12___6;
    idx12___6(0)=true;
    idx12___6(1)=true;
    idx12___6(2)=false;
    idx12___6(3)=false;
    idx12___6(4)=false;
    idx12___6(5)=true;
    cf12___6.SetIdx(idx12___6);

    vnl_vector_fixed<bool,6> idx12_456;
    idx12_456(0)=true;
    idx12_456(1)=true;
    idx12_456(2)=false;
    idx12_456(3)=true;
    idx12_456(4)=true;
    idx12_456(5)=true;
    cf12_456.SetIdx(idx12_456);

    vnl_vector_fixed<bool,6> idx123456;
    idx123456(0)=true;
    idx123456(1)=true;
    idx123456(2)=true;
    idx123456(3)=true;
    idx123456(4)=true;
    idx123456(5)=true;
    cf123456.SetIdx(idx123456);


    vnl_vector_fixed<float,6> init;
    init(0)=0.0;init(1)=0.0;init(2)=0.0;
    init(3)=0.0;init(4)=0.0;init(5)=0.0;
    cf12___6.SetInit(init);
    cf12_456.SetInit(init);
    cf123456.SetInit(init);


    vnl_vector_fixed<float,6> offsetL;
    vnl_vector_fixed<float,6> offsetS;

    offsetL(0)=2.5;
    offsetL(1)=2.5;
    offsetL(2)=2.5;
    offsetL(3)=3.14/4;
    offsetL(4)=3.14/4;
    offsetL(5)=3.14/4;
    offsetS(0)=0.5;
    offsetS(1)=0.5;
    offsetS(2)=0.5;
    offsetS(3)=3.14/16;
    offsetS(4)=3.14/16;
    offsetS(5)=3.14/16;


    cf12___6.SetOffset(offsetL);
    cf12_456.SetOffset(offsetL);
    cf123456.SetOffset(offsetS);


    std::vector<FloatMatrixType> logs;
    FloatMatrixType log;

    std::cout << "Minimizing using method " << method << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    ///////////////////////////

    if( method == 1)
    {
        log = Amoeba(cf12___6, 100); logs.push_back(log);
        cf12_456.SetInit(cf12___6.GetTransform());
        log = Amoeba(cf12_456, 500); logs.push_back(log);
        cf123456.SetInit(cf12_456.GetTransform());
        log = Amoeba(cf123456, 500); logs.push_back(log);
    }
    else if(method == 2)
    {
        log = Powell(cf12___6,stepsizes[0]); logs.push_back(log);
        cf12_456.SetInit(cf12___6.GetTransform());
        log = Powell(cf12_456,stepsizes[1]); logs.push_back(log);
        cf123456.SetInit(cf12_456.GetTransform());
        log = Powell(cf123456,stepsizes[2]); logs.push_back(log);
    }
    else if(method == 3)
    {
        log = LBFGS(cf12___6); logs.push_back(log);
        cf12_456.SetInit(cf12___6.GetTransform());
        log = LBFGS(cf12_456); logs.push_back(log);
        cf123456.SetInit(cf12_456.GetTransform());
        log = LBFGS(cf123456); logs.push_back(log);
    }
    //////////////////////////
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time (ms): " << milliseconds << std::endl;

    std::cout << "Final Transform: ";
    for(int k=0; k<6; k++)
    {
        std::cout << cf123456.GetTransform()[k] << " ";
    }
    std::cout << std::endl;

    FloatMatrixType allLog;
    for( int k=0; k < logs.size(); k++)
    {
        log = logs[k];
        for( int m=0; m < log.size(); m++)
        {
            allLog.push_back(log[m]);
        }
    }

    std::vector<float> finalTransform;
    vnl_vector_fixed<float,6> ft = cf123456.GetTransform();
    for( int k=0; k<6; k++ )
    {
        finalTransform.push_back(ft[k]);
    }

    allLog.push_back(finalTransform);

    std::vector<float> timing;
    timing.push_back(milliseconds);
    allLog.push_back(timing);

    return allLog;

}


FloatMatrixType Amoeba(PatchRayCastCostFunction &cf, int iter)
{
    vnl_amoeba Minimizer(cf);

    Minimizer.set_f_tolerance(0.0001);
    Minimizer.set_x_tolerance(0.001);
    Minimizer.set_max_iterations (iter);

    vnl_vector<double> startingVector = cf.ComputeStartingVector();
    Minimizer.minimize(startingVector);

    return cf.GetHistory();
}

FloatMatrixType LBFGS(PatchRayCastCostFunction &cf)
{
    vnl_lbfgs Minimizer(cf);

    Minimizer.set_f_tolerance(0.0001);
    Minimizer.set_x_tolerance(0.001);

    vnl_vector<double> startingVector = cf.ComputeStartingVector();
    Minimizer.minimize(startingVector);

    return cf.GetHistory();
}

FloatMatrixType Powell(PatchRayCastCostFunction &cf, double stepsize)
{
    vnl_powell Minimizer(&cf);

    Minimizer.set_initial_step(stepsize);
    Minimizer.set_f_tolerance(0.0001);
    Minimizer.set_x_tolerance(0.001);

    vnl_vector<double> startingVector = cf.ComputeStartingVector();
    Minimizer.minimize(startingVector);

    return cf.GetHistory();
}

void WriteHistory(std::string logFileName, FloatMatrixType optimizationLog)
{
    std::ofstream outfile(logFileName.c_str());
    if(!outfile){printf("Couldn't open file: %s\n",logFileName.c_str()); ;return;}
    for( int i=0; i<optimizationLog.size(); i++)
    {
        for( int j=0; j<optimizationLog[i].size();j++)
        {
            outfile << optimizationLog[i][j] << "\t";
        }
        outfile << "\n";
    }
    outfile.close();
}

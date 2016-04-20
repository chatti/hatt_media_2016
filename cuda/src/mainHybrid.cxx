#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include "DASH.h"
#include "DASHCostFunction.h"
#include "PatchRayCast.h"
#include "PatchRayCastCostFunction.h"
#include <vnl/algo/vnl_lbfgs.h>  //limited memory BFGS algorithm (general unconstrained optimization)
#include <vnl/algo/vnl_lbfgsb.h> //constrained BFGS
#include <vnl/algo/vnl_amoeba.h> //Nelder-mead
#include <vnl/algo/vnl_powell.h> //Powell
FloatMatrixType Amoeba_DASH(DASHCostFunction &cf,int iter);
FloatMatrixType Powell_DASH(DASHCostFunction &cf,double stepsize);
FloatMatrixType LBFGS_DASH(DASHCostFunction &cf);

FloatMatrixType Amoeba_PRC(PatchRayCastCostFunction &cf,int iter);
FloatMatrixType Powell_PRC(PatchRayCastCostFunction &cf,double stepsize);
FloatMatrixType LBFGS_PRC(PatchRayCastCostFunction &cf);

FloatMatrixType Biplane(  DASH &X, PatchRayCast &Y, int method);
FloatMatrixType Monoplane(DASH &X, PatchRayCast &Y, int method);
void WriteHistory(std::string logFileName, FloatMatrixType optimizationLog);

int main(int argc, char* args[] ){

    if(argc < 6)
    {
        printf("Proper usage: dash [parameterfile1] [parameterfile2] [logfile] [Metric (1-2)] [OptMethod (1-3)]\n");
        return -1;
    }

    std::string pfileDASH;
    std::string pfilePRC;
    std::string logfile;
    int method = 0;
    int metric = 0;

    std::stringstream ss;
    ss << args[1]; ss >> pfileDASH; ss.str(""); ss.clear();
    ss << args[2]; ss >> pfilePRC;  ss.str(""); ss.clear();
    ss << args[3]; ss >> logfile;   ss.str(""); ss.clear();
    ss << args[4]; ss >> metric;    ss.str(""); ss.clear();
    ss << args[5]; ss >> method;    ss.str(""); ss.clear();

    printf("Initializing DASH\n\n");
    DASH X;
    X.ReadParameterFile(pfileDASH);
    if(!X.isOK()){
        return -1;
    }
    X.SetMetric(metric);

    printf("Initializing PRC\n\n");
    PatchRayCast Y;
    Y.ReadParameterFile(pfilePRC);
    if(!Y.isOK()){
        return -1;
    }


    FloatMatrixType allLog;
    if(X.GetNumXray() == 1)
    {
        allLog = Monoplane(X,Y,method);
    }
    else
    {
        allLog = Biplane(X,Y,method);
    }

    WriteHistory(logfile, allLog);
    printf("Registration Finished\n");
    return 0;
}

FloatMatrixType Biplane(DASH &X, PatchRayCast &Y, int method)
{
    DASHCostFunction cf12XXX6(3);cf12XXX6.SetDASH(&X);
    DASHCostFunction cf12X456(5);cf12X456.SetDASH(&X);
    DASHCostFunction cf123456(6);cf123456.SetDASH(&X);

    printf("Cost Function Initialized\n");

    vnl_vector_fixed<bool,6> idx12XXX6;
    idx12XXX6(0)=true;
    idx12XXX6(1)=true;
    idx12XXX6(2)=false;
    idx12XXX6(3)=false;
    idx12XXX6(4)=false;
    idx12XXX6(5)=true;
    cf12XXX6.SetIdx(idx12XXX6);

    vnl_vector_fixed<bool,6> idx12X456;
    idx12X456(0)=true;
    idx12X456(1)=true;
    idx12X456(2)=false;
    idx12X456(3)=true;
    idx12X456(4)=true;
    idx12X456(5)=true;
    cf12X456.SetIdx(idx12X456);

    vnl_vector_fixed<bool,6> idx123456;
    idx123456(0)=true;
    idx123456(1)=true;
    idx123456(2)=false;
    idx123456(3)=false;
    idx123456(4)=true;
    idx123456(5)=true;
    cf123456.SetIdx(idx123456);


    vnl_vector_fixed<float,6> init;
    init(0)=0.0;init(1)=0.0;init(2)=0.0;
    init(3)=0.0;init(4)=0.0;init(5)=0.0;
    cf12XXX6.SetInit(init);
    cf12X456.SetInit(init);
    cf123456.SetInit(init);


    vnl_vector_fixed<float,6> offsetL;
    vnl_vector_fixed<float,6> offsetS;
    if(method ==1)
    {
        offsetL(0)=0.0;
        offsetL(1)=0.0;
        offsetL(2)=0.0;
        offsetL(3)=3.14*0;
        offsetL(4)=3.14*0;
        offsetL(5)=3.14*0;
        offsetS(0)=2.5;
        offsetS(1)=2.5;
        offsetS(2)=2.5;
        offsetS(3)=1.57*0;
        offsetS(4)=1.57*0;
        offsetS(5)=1.57*0;
    }
    else
    {
        offsetL(0)=0.0;
        offsetL(1)=0.0;
        offsetL(2)=0.0;
        offsetL(3)=0.0;
        offsetL(4)=0.0;
        offsetL(5)=0.0;
        offsetS(0)=0.0;
        offsetS(1)=0.0;
        offsetS(2)=0.0;
        offsetS(3)=0.0;
        offsetS(4)=0.0;
        offsetS(5)=0.0;
    }

    cf12XXX6.SetOffset(offsetL);
    cf12X456.SetOffset(offsetL);
    cf123456.SetOffset(offsetS);


    vector<FloatMatrixType> logs;
    FloatMatrixType log;

    cout << "Minimizing using method " << method << endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    ///////////////////////////

    if( method == 1)
    {
        log = Amoeba_DASH(cf12XXX6, 100); logs.push_back(log);
        cf12X456.SetInit(cf12XXX6.GetTransform());
        log = Amoeba_DASH(cf12X456, 250); logs.push_back(log);
        cf123456.SetInit(cf12X456.GetTransform());
        log = Amoeba_DASH(cf123456, 1000); logs.push_back(log);
    }
    else if(method == 2)
    {
        log = Powell_DASH(cf12XXX6, 2.5); logs.push_back(log);
        cf12X456.SetInit(cf12XXX6.GetTransform());
        log = Powell_DASH(cf12X456, 0.025); logs.push_back(log);
        cf123456.SetInit(cf12X456.GetTransform());
        log = Powell_DASH(cf123456, 0.010); logs.push_back(log);
    }
    else if(method == 3)
    {
        log = LBFGS_DASH(cf12XXX6); logs.push_back(log);
        cf12X456.SetInit(cf12XXX6.GetTransform());
        log = LBFGS_DASH(cf12X456); logs.push_back(log);
        cf123456.SetInit(cf12X456.GetTransform());
        log = LBFGS_DASH(cf123456); logs.push_back(log);
    }
    //////////////////////////
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time (ms): " << milliseconds << endl;


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


FloatMatrixType Monoplane(DASH &X, PatchRayCast &Y, int method)
{
    DASHCostFunction         CF1(3);CF1.SetDASH(&X);
    DASHCostFunction         CF2(5);CF2.SetDASH(&X);
    PatchRayCastCostFunction CF3(5);CF3.SetPatchRayCast(&Y);
    PatchRayCastCostFunction CF4(6);CF4.SetPatchRayCast(&Y);

    printf("Cost Function Initialized\n");

    vnl_vector_fixed<bool,6> IDX1;
    IDX1(0)=true;
    IDX1(1)=true;
    IDX1(2)=false;
    IDX1(3)=false;
    IDX1(4)=false;
    IDX1(5)=true;
    CF1.SetIdx(IDX1);

    vnl_vector_fixed<bool,6> IDX2;
    IDX2(0)=true;
    IDX2(1)=true;
    IDX2(2)=false;
    IDX2(3)=true;
    IDX2(4)=true;
    IDX2(5)=true;
    CF2.SetIdx(IDX2);

    vnl_vector_fixed<bool,6> IDX3;
    IDX3(0)=true;
    IDX3(1)=true;
    IDX3(2)=false;
    IDX3(3)=true;
    IDX3(4)=true;
    IDX3(5)=true;
    CF3.SetIdx(IDX3);

    vnl_vector_fixed<bool,6> IDX4;
    IDX4(0)=true;
    IDX4(1)=true;
    IDX4(2)=true;
    IDX4(3)=true;
    IDX4(4)=true;
    IDX4(5)=true;
    CF4.SetIdx(IDX4);

    vnl_vector_fixed<float,6> init;
    init(0)=0.0;
    init(1)=0.0;
    init(2)=0.0;
    init(3)=0.0;
    init(4)=0.0;
    init(5)=0.0;
    CF1.SetInit(init);
    CF2.SetInit(init);
    CF3.SetInit(init);
    CF4.SetInit(init);

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

    CF1.SetOffset(offsetL);
    CF2.SetOffset(offsetL);
    CF3.SetOffset(offsetL);
    CF4.SetOffset(offsetS);

    vector<FloatMatrixType> logs;
    FloatMatrixType log;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    ///////////////////////////


    if(method == 1)
    {
        log = Amoeba_DASH(CF1,  250); logs.push_back(log);
        CF2.SetInit(CF1.GetTransform());
        log = Amoeba_DASH(CF2,  250); logs.push_back(log);
        CF3.SetInit(CF2.GetTransform());
        log = Amoeba_PRC(CF3,  250); logs.push_back(log);
        CF4.SetInit(CF3.GetTransform());
        //log = Amoeba_PRC(CF4,  750); logs.push_back(log);
    }
    else if( method == 2)
    {
        log = Powell_DASH(CF1,  2.5); logs.push_back(log);
        CF2.SetInit(CF1.GetTransform());
        log = Powell_DASH(CF2,  0.5); logs.push_back(log);
        CF3.SetInit(CF2.GetTransform());
        log = Powell_PRC(CF3,  0.5); logs.push_back(log);
        CF4.SetInit(CF3.GetTransform());
        //log = Powell_PRC(CF4,  0.05); logs.push_back(log);
    }
    else if( method ==3)
    {
        log = LBFGS_DASH(CF1); logs.push_back(log);
        CF2.SetInit(CF1.GetTransform());
        log = LBFGS_DASH(CF2); logs.push_back(log);
        CF3.SetInit(CF2.GetTransform());
        log = LBFGS_PRC(CF3); logs.push_back(log);
        CF4.SetInit(CF3.GetTransform());
        //log = LBFGS_PRC(CF4); logs.push_back(log);
    }

    //////////////////////////
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time (ms): " << milliseconds << endl;


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
    vnl_vector_fixed<float,6> ft = CF3.GetTransform();
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


FloatMatrixType Amoeba_DASH(DASHCostFunction &cf, int iter)
{
    vnl_amoeba Minimizer(cf);

    Minimizer.set_f_tolerance(0.001);
    Minimizer.set_x_tolerance(0.001);
    Minimizer.set_max_iterations (iter);

    vnl_vector<double> startingVector = cf.ComputeStartingVector();
    Minimizer.minimize(startingVector);


	cout << "NumEvals: " << Minimizer.get_num_evaluations() << endl;
    return cf.GetHistory();
}

FloatMatrixType LBFGS_DASH(DASHCostFunction &cf)
{
    vnl_lbfgs Minimizer(cf);

    Minimizer.set_f_tolerance(0.001);
    Minimizer.set_x_tolerance(0.001);

    vnl_vector<double> startingVector = cf.ComputeStartingVector();
    Minimizer.minimize(startingVector);

    cout << "NumEvals: " << Minimizer.get_num_evaluations() << endl;
    return cf.GetHistory();
}

FloatMatrixType Powell_DASH(DASHCostFunction &cf, double stepsize)
{
    vnl_powell Minimizer(&cf);

    Minimizer.set_initial_step(stepsize);
    Minimizer.set_f_tolerance(0.001);
    Minimizer.set_x_tolerance(0.001);
    //Minimizer.set_max_iterations (500);

    vnl_vector<double> startingVector = cf.ComputeStartingVector();
    Minimizer.minimize(startingVector);


    cout << "NumEvals: " << Minimizer.get_num_evaluations() << endl;
    return cf.GetHistory();
}

FloatMatrixType Amoeba_PRC(PatchRayCastCostFunction &cf, int iter)
{
    vnl_amoeba Minimizer(cf);

    Minimizer.set_f_tolerance(0.0001);
    Minimizer.set_x_tolerance(0.001);
    Minimizer.set_max_iterations (iter);

    vnl_vector<double> startingVector = cf.ComputeStartingVector();
    Minimizer.minimize(startingVector);


    cout << "NumEvals: " << Minimizer.get_num_evaluations() << endl;
    return cf.GetHistory();
}

FloatMatrixType LBFGS_PRC(PatchRayCastCostFunction &cf)
{
    vnl_lbfgs Minimizer(cf);

    Minimizer.set_f_tolerance(0.0001);
    Minimizer.set_x_tolerance(0.001);

    vnl_vector<double> startingVector = cf.ComputeStartingVector();
    Minimizer.minimize(startingVector);

    cout << "NumEvals: " << Minimizer.get_num_evaluations() << endl;
    return cf.GetHistory();
}

FloatMatrixType Powell_PRC(PatchRayCastCostFunction &cf, double stepsize)
{
    vnl_powell Minimizer(&cf);

    Minimizer.set_initial_step(stepsize);
    Minimizer.set_f_tolerance(0.0001);
    Minimizer.set_x_tolerance(0.001);
    //Minimizer.set_max_iterations (500);

    vnl_vector<double> startingVector = cf.ComputeStartingVector();
    Minimizer.minimize(startingVector);


    cout << "NumEvals: " << Minimizer.get_num_evaluations() << endl;
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


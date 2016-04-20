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
#include <vnl/algo/vnl_lbfgs.h>  //limited memory BFGS algorithm (general unconstrained optimization)
#include <vnl/algo/vnl_lbfgsb.h> //constrained BFGS
#include <vnl/algo/vnl_amoeba.h> //Nelder-mead
#include <vnl/algo/vnl_powell.h> //Powell
FloatMatrixType Amoeba(DASHCostFunction &cf,int iter);
FloatMatrixType Powell(DASHCostFunction &cf,double stepsize);
FloatMatrixType LBFGS(DASHCostFunction &cf);
FloatMatrixType Biplane(DASH &X, int method);
FloatMatrixType Monoplane(DASH &X, int method);
void WriteHistory(std::string logFileName, FloatMatrixType optimizationLog);

int main(int argc, char* args[] ){

    if(argc < 5)
    {
        printf("Proper usage: dash [parameterfile] [logfile] [Metric (1-2)] [OptMethod (1-3)]\n");
        return -1;
    }

    std::string pfile;
    std::string logfile;
    int method = 0;
    int metric = 0;

    std::stringstream ss;
    ss << args[1]; ss >> pfile;   ss.str(""); ss.clear();
    ss << args[2]; ss >> logfile; ss.str(""); ss.clear();
    ss << args[3]; ss >> metric;  ss.str(""); ss.clear();
    ss << args[4]; ss >> method;  ss.str(""); ss.clear();


    DASH X;
    X.ReadParameterFile(pfile);
    if(!X.isOK()){
        return -1;
    }

    X.SetMetric(metric);

    FloatMatrixType allLog;
    if(X.GetNumXray() == 1)
    {
        allLog = Monoplane(X,method);
    }
    else
    {
        allLog = Biplane(X,method);
    }

    WriteHistory(logfile, allLog);
    printf("Registration Finished\n");
    return 0;
}

FloatMatrixType Biplane(DASH &X,int method)
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
        log = Amoeba(cf12XXX6, 100); logs.push_back(log);
        cf12X456.SetInit(cf12XXX6.GetTransform());
        log = Amoeba(cf12X456, 250); logs.push_back(log);
        cf123456.SetInit(cf12X456.GetTransform());
        log = Amoeba(cf123456, 1000); logs.push_back(log);
    }
    else if(method == 2)
    {
        log = Powell(cf12XXX6, 1); logs.push_back(log);
        cf12X456.SetInit(cf12XXX6.GetTransform());
        log = Powell(cf12X456, 0.1); logs.push_back(log);
        cf123456.SetInit(cf12X456.GetTransform());
        log = Powell(cf123456, 0.1); logs.push_back(log);
    }
    else if(method == 3)
    {
        log = LBFGS(cf12XXX6); logs.push_back(log);
        cf12X456.SetInit(cf12XXX6.GetTransform());
        log = LBFGS(cf12X456); logs.push_back(log);
        cf123456.SetInit(cf12X456.GetTransform());
        log = LBFGS(cf123456); logs.push_back(log);
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


FloatMatrixType Monoplane(DASH &X, int method)
{
    DASHCostFunction cf12XXX6(3);cf12XXX6.SetDASH(&X);
    DASHCostFunction cf12X456(5);cf12X456.SetDASH(&X);
    DASHCostFunction cf123456(6);cf123456.SetDASH(&X);

    printf("Cost Function Initialized\n");

    vnl_vector_fixed<bool,6> idx12XXX6;
    idx12XXX6(0)=true;idx12XXX6(1)=true;idx12XXX6(2)=false;
    idx12XXX6(3)=false;idx12XXX6(4)=false;idx12XXX6(5)=true;
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
    cf12XXX6.SetInit(init);
    cf12X456.SetInit(init);
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

    cf12XXX6.SetOffset(offsetL);
    cf12X456.SetOffset(offsetL);
    cf123456.SetOffset(offsetS);


    vector<FloatMatrixType> logs;
    FloatMatrixType log;

    //cout << "Minimizing using Amoeba" << endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    ///////////////////////////


    if(method == 1)
    {
        log = Amoeba(cf12XXX6,  500); logs.push_back(log);
        cf12X456.SetInit(cf12XXX6.GetTransform());
        log = Amoeba(cf12X456,  1500); logs.push_back(log);
        //cf123456.SetInit(cf12X456.GetTransform());
        //log = Amoeba(cf123456, 500); logs.push_back(log);
    }
    else if( method == 2)
    {
        log = Powell(cf12XXX6, 2.5); logs.push_back(log);
        cf12X456.SetInit(cf12XXX6.GetTransform());
        log = Powell(cf12X456, 0.5); logs.push_back(log);
        //cf123456.SetInit(cf12X456.GetTransform());
        //log = Powell(cf123456,0.01); logs.push_back(log);
    //cf123456.f((const vnl_vector<double>)cf123456.ComputeStartingVector());
    }
    else if( method ==3)
    {
        log = LBFGS(cf12XXX6); logs.push_back(log);
        cf12X456.SetInit(cf12XXX6.GetTransform());
        log = LBFGS(cf12X456); logs.push_back(log);
        //cf123456.SetInit(cf12X456.GetTransform());
        //log = LBFGS(cf123456); logs.push_back(log);
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



FloatMatrixType Amoeba(DASHCostFunction &cf, int iter)
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

FloatMatrixType LBFGS(DASHCostFunction &cf)
{
    vnl_lbfgs Minimizer(cf);

    Minimizer.set_f_tolerance(0.0001);
    Minimizer.set_x_tolerance(0.001);

    vnl_vector<double> startingVector = cf.ComputeStartingVector();
    Minimizer.minimize(startingVector);

    cout << "NumEvals: " << Minimizer.get_num_evaluations() << endl;
    return cf.GetHistory();
}

FloatMatrixType Powell(DASHCostFunction &cf, double stepsize)
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

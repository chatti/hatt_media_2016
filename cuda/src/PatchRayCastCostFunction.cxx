#include "PatchRayCastCostFunction.h"
#include <vnl/vnl_vector.h>
#include <vnl/vnl_vector_fixed.h>
#include <vnl/vnl_cost_function.h>

double PatchRayCastCostFunction::f(vnl_vector<double> const &x)
{

    double returnValue = 0.0;

    int i=0;
    for( int k=0;k< 6; k++)
    {
        if(idx(k)){
            transform(k) = (float)x(i)-offset(k)+init(k);
            i++;
        }
        else{
            transform(k) = init(k);
        }
    }

    X->SetSecondaryTransform(transform(0),transform(1),transform(2),transform(3),transform(4),transform(5));
    returnValue = (double)X->ComputePatchGCC();

    if(recordHistory){
        std::vector<float> tmp;
        for(int k=0;k<6; k++){
            tmp.push_back(transform(k));
        }
        tmp.push_back(returnValue);
        //std::cout << tmp << std::endl;
        history.push_back(tmp);
    }

    return returnValue;

}

void PatchRayCastCostFunction::gradf(vnl_vector<double> const &x, vnl_vector<double> &dx)
{
    //OR, use Finite Difference gradient
    fdgradf(x, dx);
}

void PatchRayCastCostFunction::SetPatchRayCast(PatchRayCast* inptr)
{
    X = inptr;
}


void PatchRayCastCostFunction::SetInit(vnl_vector_fixed<float,6> inputInit)
{
    init = inputInit;
}

vnl_vector_fixed<float,6> PatchRayCastCostFunction::GetInit()
{
    return init;
}

void PatchRayCastCostFunction::SetIdx(vnl_vector_fixed<bool,6> inputIdx)
{
    idx = inputIdx;
}

vnl_vector_fixed<bool,6> PatchRayCastCostFunction::GetIdx()
{
    return idx;
}

void PatchRayCastCostFunction::SetOffset(vnl_vector<float> inputOffset)
{
    offset = inputOffset;
}

vnl_vector_fixed<float,6> PatchRayCastCostFunction::GetOffset()
{
    return offset;
}

vnl_vector_fixed<float,6> PatchRayCastCostFunction::GetTransform()
{
    return transform;
}

vnl_vector<double> PatchRayCastCostFunction::ComputeStartingVector()
{

    vnl_vector<double> startingVector(numParams);

    int i=0;
    for(int k=0;k< 6; k++){
        if(idx(k)){
            startingVector(i) = (double)offset(k);
            i++;
        }
    }
    return startingVector;
}

FloatMatrixType PatchRayCastCostFunction::GetHistory()
{
    return history;
}




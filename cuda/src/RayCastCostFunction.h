#include "RayCast.h"
#include <vnl/vnl_vector.h>
#include <vnl/vnl_vector_fixed.h>
#include <vnl/vnl_cost_function.h>

using namespace std;

typedef std::vector<std::vector<float> > FloatMatrixType;

class RayCastCostFunction : public vnl_cost_function
{
public:


    RayCast* X;
    FloatMatrixType history;
    bool recordHistory;
    vnl_vector_fixed<float,6>  init;
    vnl_vector_fixed<float,6>  offset;
    vnl_vector_fixed<float, 6>  transform;
    vnl_vector_fixed<bool,6>    idx;
    vnl_vector<double>  x0;
    bool gradient;

    int numParams;

    RayCastCostFunction(const int NumVars) : vnl_cost_function(NumVars)
    {
        numParams = (int)NumVars;
        recordHistory=true;
        gradient=false;
    }

    double f(vnl_vector<double> const &x)
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

        if(!gradient)
        {
            returnValue = (double)X->ComputeNCC();
        }
        else if(gradient)
        {
            returnValue = (double)X->ComputeNGC();
        }

        //printf("%f\n",returnValue);


        if(recordHistory){
            std::vector<float> tmp;
            for(int k=0;k<6; k++){
                tmp.push_back(transform(k));
            }
            tmp.push_back(returnValue);
            history.push_back(tmp);
        }
        return returnValue;
    }

    void gradf(vnl_vector<double> const &x, vnl_vector<double> &dx)
    {
        //OR, use Finite Difference gradient
        fdgradf(x, dx);
    }

    void SetRayCast(RayCast* inptr)
    {
        X = inptr;
    }

    void SetGradient(bool usegradient)
    {
        gradient = usegradient;
    }

    void SetInit(vnl_vector_fixed<float,6> inputInit)
    {
        init = inputInit;
    }

    vnl_vector_fixed<float,6> GetInit()
    {
        return init;
    }

    void SetIdx(vnl_vector_fixed<bool,6> inputIdx)
    {
        idx = inputIdx;
    }

    vnl_vector_fixed<bool,6> GetIdx()
    {
        return idx;
    }

    void SetOffset(vnl_vector<float> inputOffset)
    {
        offset = inputOffset;
    }

    vnl_vector_fixed<float,6> GetOffset()
    {
        return offset;
    }

    vnl_vector_fixed<float,6> GetTransform()
    {
        return transform;
    }

    vnl_vector<double> ComputeStartingVector()
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

    FloatMatrixType GetHistory()
    {
        return history;
    }


};


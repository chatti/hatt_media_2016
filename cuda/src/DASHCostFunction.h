#include "DASH.h"
#include <vnl/vnl_vector.h>
#include <vnl/vnl_vector_fixed.h>
#include <vnl/vnl_cost_function.h>

using namespace std;

typedef std::vector<std::vector<float> > FloatMatrixType;

class DASHCostFunction : public vnl_cost_function
{
public:


    DASH* X;
    FloatMatrixType history;
    bool recordHistory;
    vnl_vector_fixed<float,6>  init;
    vnl_vector_fixed<float,6>  offset;
    vnl_vector_fixed<float, 6>  transform;
    vnl_vector_fixed<bool,6>    idx;
    vnl_vector<double>  x0;

	int numParams;
	
    DASHCostFunction(const int NumVars) : vnl_cost_function(NumVars)
	{
		numParams = (int)NumVars;
        recordHistory=true;
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


        X->SetTransform(transform(0),transform(1),transform(2),transform(3),transform(4),transform(5));
        X->UpdateTransformMatrix();
        returnValue = (double)X->ComputeDASHCostFunction();

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

    void SetDASH(DASH* inptr)
    {
        X = inptr;
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


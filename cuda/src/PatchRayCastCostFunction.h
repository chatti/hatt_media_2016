#ifndef PATCHRAYCASTCOSTFUNCTION_H
#define PATCHRAYCASTCOSTFUNCTION_H

#include <vector>
#include "PatchRayCast.h"
#include <vnl/vnl_vector.h>
#include <vnl/vnl_vector_fixed.h>
#include <vnl/vnl_cost_function.h>

//using namespace std;
typedef std::vector<std::vector<float> > FloatMatrixType;

class PatchRayCastCostFunction : public vnl_cost_function
{



    PatchRayCast* X;
    FloatMatrixType history;
    bool recordHistory;
    vnl_vector_fixed<float,6>  init;
    vnl_vector_fixed<float,6>  offset;
    vnl_vector_fixed<float, 6>  transform;
    vnl_vector_fixed<bool,6>    idx;
    vnl_vector<double>  x0;
    int numParams;

    public:

        PatchRayCastCostFunction(const int NumVars) : vnl_cost_function(NumVars)
        {
            numParams = (int)NumVars;
            recordHistory=true;
        }
        double f(vnl_vector<double> const &x);
        void gradf(vnl_vector<double> const &x, vnl_vector<double> &dx);
        void SetPatchRayCast(PatchRayCast* inptr);
        void SetInit(vnl_vector_fixed<float,6> inputInit);
        vnl_vector_fixed<float,6> GetInit();
        void SetIdx(vnl_vector_fixed<bool,6> inputIdx);
        vnl_vector_fixed<bool,6> GetIdx();
        void SetOffset(vnl_vector<float> inputOffset);
        vnl_vector_fixed<float,6> GetOffset();
        vnl_vector_fixed<float,6> GetTransform();
        vnl_vector<double> ComputeStartingVector();
        FloatMatrixType GetHistory();

};

#endif

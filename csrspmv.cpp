#include <iostream>
#include <string>
#include <cstdlib>
#include <cuda_runtime.h>
#include "cusparse.h"
#include "mvstructs.h"
/*
define CLEANUP(s)                                   \
do {                                                 \
    printf ("%s\n", s);                              \
    if (yHostPtr)           free(yHostPtr);          \
    if (zHostPtr)           free(zHostPtr);          \
    if (xIndHostPtr)        free(xIndHostPtr);       \
    if (xValHostPtr)        free(xValHostPtr);       \
    if (cooRowIndexHostPtr) free(cooRowIndexHostPtr);\
    if (cooColIndexHostPtr) free(cooColIndexHostPtr);\
    if (cooValHostPtr)      free(cooValHostPtr);     \
    if (y)                  cudaFree(y);             \
    if (z)                  cudaFree(z);             \
    if (xInd)               cudaFree(xInd);          \
    if (xVal)               cudaFree(xVal);          \
    if (csrRowPtr)          cudaFree(csrRowPtr);     \
    if (cooRowIndex)        cudaFree(cooRowIndex);   \
    if (cooColIndex)        cudaFree(cooColIndex);   \
    if (cooVal)             cudaFree(cooVal);        \
    if (descr)              cusparseDestroyMatDescr(descr);\
    if (handle)             cusparseDestroy(handle); \
    cudaDeviceReset();          \
    fflush (stdout);                                 \
} while (0)*/


int main(int argc, char const *argv[])
{
    // build the matrix vector
    cooMatrix cm(9, 4);
    cm.push(0,0,1.0);
    cm.push(2,0,2.0);
    cm.push(3,0,3.0);
    cm.push(1,1,4.0);
    cm.push(0,2,5.0);
    cm.push(2,2,6.0);
    cm.push(3,2,7.0);
    cm.push(1,3,8.0);
    cm.push(3,3,9.0);
    cm.print();

    denseVector v(8);
    v.push(10.0);
    v.push(20.0);
    v.push(30.0);
    v.push(40.0);
    v.push(50.0);
    v.push(60.0);
    v.push(70.0);
    v.push(80.0);
    v.print();

    denseVector result(8);
    

    
    cusparseStatus_t status;
    cusparseHandle_t handle=0;

    // init cusparse
    status = cusparseCreate(&handle);
    if(status != CUSPARSE_STATUS_SUCCESS){
        std::cout<<"CUSPARSE Library initialisation failed"<<std::endl;
        cusparseDestroy(handle);
        return 1;
    }

    // create a matrix descriptor
    cusparseMatDescr_t descr=0;
    status = cusparseCreateMatDescr(&descr);
    if(status != CUSPARSE_STATUS_SUCCESS) {
        std::cout<<"Matrix desriptor initialisation failed"<<std::endl;
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(handle);
        return 1;
    }
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    csrMatrix csrm = cm.asCSR(handle);

    return 0;
}
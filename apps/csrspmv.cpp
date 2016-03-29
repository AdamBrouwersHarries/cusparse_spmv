#include <iostream>
#include <string>
#include <cstdlib>
#include <cuda_runtime.h>
#include "cusparse.h"
#include "mvstructs.h"

cusparseHandle_t initCUSparse() {
    cusparseStatus_t status;
    cusparseHandle_t handle=0;

    // init cusparse
    status = cusparseCreate(&handle);
    if(status != CUSPARSE_STATUS_SUCCESS){
        std::cout<<"CUSPARSE Library initialisation failed"<<std::endl;
        cusparseDestroy(handle);
        exit(1);
    }

    // get ad device, and check it has compute capability 1.3
    int devID;
    cudaDeviceProp prop;
    cudaError_t cudaStat;
    cudaStat = cudaGetDevice(&devID);
    if(cudaSuccess != cudaStat){
        std::cout<<"Error: cudaGetDevice failed!"<<std::endl;
        // do some cleanup...
        cusparseDestroy(handle);
        std::cout<<"Error: cudaStat: "<<cudaStat<<", "<<
            cudaGetErrorString(cudaStat)<<std::endl;
        exit(1);
    }

    cudaStat = cudaGetDeviceProperties( &prop, devID) ;
    if (cudaSuccess != cudaStat){
        std::cout<<"Error: cudaGetDeviceProperties failed!"<<std::endl;
        // do some cleanup...
        cusparseDestroy(handle);
        std::cout<<"Error: cudaStat: "<<cudaStat<<", "<<
            cudaGetErrorString(cudaStat)<<std::endl;
        exit(1);
    }

    int cc = 100*prop.major + 10*prop.minor;
    if (cc <= 130){
        cusparseDestroy(handle);

        std::cout<<"waive the test because only sm13 and above are supported"<<std::endl;
        std::cout<<"the device has compute capability"<<cc<<std::endl;
        std::cout<<"example test WAIVED"<<std::endl;
        exit(2);
    }else{
        std::cout<<"Compute capability: "<<prop.major<<"."<<prop.minor<<std::endl;
    }

    return handle;
}


int main(int argc, char const *argv[])
{
    // build the matrix vector
    cooMatrix cm(9, 4, 4);
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
    
    cusparseHandle_t handle = initCUSparse();

    csrMatrix csrm = cm.asCSR(handle);

    // compute a sparse matrix vector multiplication
    csrm.spmv(handle, v, result);

    std::cout<<"result after: "<<std::endl;
    result.print();

    cusparseDestroy(handle);

    return 0;
}
#include <iostream>
#include <string>
#include <cstdlib>
#include <tuple>
#include <cuda_runtime.h>

#include "cusparse.h"
#include "mvstructs.h"

std::tuple<cusparseHandle_t, std::string> initCUSparse() {
    cusparseStatus_t status;
    cusparseHandle_t handle=0;

    // init cusparse
    status = cusparseCreate(&handle);
    if(status != CUSPARSE_STATUS_SUCCESS){
        std::cerr<<"CUSPARSE Library initialisation failed"<<std::endl;
        cusparseDestroy(handle);
        exit(1);
    }

    // get ad device, and check it has compute capability 1.3
    int devID;
    cudaDeviceProp prop;
    cudaError_t cudaStat;
    cudaStat = cudaGetDevice(&devID);
    if(cudaSuccess != cudaStat){
        std::cerr<<"Error: cudaGetDevice failed!"<<std::endl;
        // do some cleanup...
        cusparseDestroy(handle);
        std::cerr<<"Error: cudaStat: "<<cudaStat<<", "<<
            cudaGetErrorString(cudaStat)<<std::endl;
        exit(1);
    }

    cudaStat = cudaGetDeviceProperties( &prop, devID) ;
    if (cudaSuccess != cudaStat){
        std::cerr<<"Error: cudaGetDeviceProperties failed!"<<std::endl;
        // do some cleanup...
        cusparseDestroy(handle);
        std::cerr<<"Error: cudaStat: "<<cudaStat<<", "<<
            cudaGetErrorString(cudaStat)<<std::endl;
        exit(1);
    }

    int cc = 100*prop.major + 10*prop.minor;
    if (cc <= 130){
        cusparseDestroy(handle);

        std::cerr<<"waive the test because only sm13 and above are supported"<<std::endl;
        std::cerr<<"the device has compute capability"<<cc<<std::endl;
        std::cerr<<"example test WAIVED"<<std::endl;
        exit(2);
    }else{
        std::cerr<<"Compute capability: "<<prop.major<<"."<<prop.minor<<std::endl;
    }

    std::string deviceName(prop.name);
    return std::make_tuple(handle, deviceName);
}

void printSqlResult(std::string host,
                    std::string device,
                    std::string matrix,
                    float runtime) {
    std::cout<<"insert into TABLE (time, host, device, matrix) values ("<<
        runtime << ", \'" <<
        host    << "\', \'" <<
        device  << "\', \'" <<
        matrix  << "\'"<<
        ");" << std::endl;
}

int main(int argc, char const *argv[])
{
    if(argc < 4){
        std::cerr<<"Error: no hostname given!"<<std::endl;
        exit(1);
    }
    if(argc < 3){
        std::cerr<<"Error: no matrix name given!"<<std::endl;
        exit(1);
    }
    if(argc < 2){
        std::cerr<<"Error: no matrix file specified!"<<std::endl;
        exit(1);
    }
    std::string mfname(argv[1]);
    std::string mname(argv[2]);
    std::string hostname(argv[3]);
    std::cerr<<"Matrix filename: "<<mfname<<std::endl;
    std::cerr<<"Matrix name: "<<mname<<std::endl;
    std::cerr<<"Hostname: "<<hostname<<std::endl;

    // input matrix
    cooMatrix cm(mfname);

    // input vector
    denseVector v(cm.getWidth());
    v.fillRandom();
    std::cerr<<"v before: "<<std::endl;
    v.print();

    // output vector
    denseVector result(cm.getWidth());
    
    // cusparse handle
    auto hn = initCUSparse();
    auto handle = std::get<0>(hn);
    auto devname = std::get<1>(hn);
    std::cerr<<"Running on device: "<<devname<<std::endl;

    // csr version of the matrix
    csrMatrix csrm = cm.asCSR(handle);

    // compute a sparse matrix vector multiplication
    auto times = csrm.spmv(handle, v, result);
    std::sort(times.begin(), times.end());
    if(times.size()%2){
        std::cerr<<"Median: "<<times[times.size()/2]<<std::endl;
    }else{
        std::cerr<<"Median: "<<times[(times.size()+1)/2]<<std::endl;
    }
    for(auto rt : times){
        printSqlResult(hostname, devname, mname + ".mtx", rt);
    }


    // the result
    std::cerr<<"result after: "<<std::endl;
    result.print();

    // clean up
    cusparseDestroy(handle);

    return 0;
}
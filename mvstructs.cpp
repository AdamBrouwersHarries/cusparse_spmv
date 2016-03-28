#include "mvstructs.h"

// =============================================================================
// =============================== CSR Matrix ==================================
// =============================================================================


csrMatrix::csrMatrix(int _nnz, int height, int* _rp, int* _cixp, float* _vp) {
    rowDevPtr = _rp;
    colIxDevPtr = _cixp;
    valDevPtr = _vp;
    nnz = _nnz;
}
csrMatrix::~csrMatrix() {
    if(rowDevPtr) cudaFree(rowDevPtr);
    if(colIxDevPtr) cudaFree(colIxDevPtr);
    if(valDevPtr) cudaFree(valDevPtr);
}

// =============================================================================
// =============================== COO Matrix ==================================
// =============================================================================

// public

cooMatrix::cooMatrix(int _nnz, int _height) {
    nnz = _nnz;
    height = _height;
    // allocate on the host
    rowIndexHostPtr = new int[nnz];
    colIndexHostPtr = new int[nnz];
    valHostPtr = new float[nnz];
    // allocate ont the device
    cudaError_t rowAlloc = cudaMalloc((void**)&rowIndexDevPtr, nnz*sizeof(int));
    cudaError_t colAlloc = cudaMalloc((void**)&colIndexDevPtr, nnz*sizeof(int));
    cudaError_t valAlloc = cudaMalloc((void**)&valDevPtr, nnz*sizeof(float));
    // check for errors
    if((rowAlloc != cudaSuccess) || 
       (colAlloc != cudaSuccess) || 
       (valAlloc != cudaSuccess)) {
        std::cout<<"Device malloc failed at line "<<__LINE__<<" in "<<__FILE__<<std::endl;
        cleanup();
    }   
    // set the indexing pointer for adding nonzero elements
    ixPtr = 0;
    // assume we start with the device buffers outdated
    mstatus = DeviceOutdated;
}
cooMatrix::~cooMatrix() {
    cleanup();
}

void cooMatrix::push(int col, int row, float val){
    if(!colIndexHostPtr || !rowIndexHostPtr || !valHostPtr){
        std::cout<<"Error: attemping to push elements before allocating."<<std::endl;
    }
    if(ixPtr >= nnz){
        std::cout<<"Error: allocating too many elements!"<<std::endl;
    }
    rowIndexHostPtr[ixPtr] = row;
    colIndexHostPtr[ixPtr] = col;
    valHostPtr[ixPtr] = val;
    ixPtr++;
}

void cooMatrix::print() {
    for (int i=0; i<nnz; i++){       
        std::cout<<"rowIndexHostPtr["<<i<<"] = "<< rowIndexHostPtr[i]<<"  ";
        std::cout<<"colIndexHostPtr["<<i<<"] = "<< colIndexHostPtr[i]<<"  ";
        std::cout<<"valHostPtr["<<i<<"] = "<<valHostPtr[i]<<std::endl;
    }
}

csrMatrix cooMatrix::asCSR(cusparseHandle_t handle) {
    updateBuffers();
    std::cout<<"Building csrmatrix"<<std::endl;
    // get a 
    int * csrRowPtr = 0;
    std::cout<<"Allocating mem"<<std::endl;

    cudaError_t alloc_s = cudaMalloc((void**)&csrRowPtr,(height+1)*sizeof(int));
    if(alloc_s != cudaSuccess) {
        std::cout<<"Device malloc failed at line "<<__LINE__<<" in "<<__FILE__<<std::endl;

        cleanup();
        cudaFree(csrRowPtr);
        exit(1);
    }
    std::cout<<"Converting"<<std::endl;

    cusparseStatus_t convert_s = cusparseXcoo2csr(handle, rowIndexDevPtr, nnz, height,
                                           csrRowPtr, CUSPARSE_INDEX_BASE_ZERO);
    if(convert_s != CUSPARSE_STATUS_SUCCESS){
        std::cout<<"Conversion from COO to CSR format failed"<<std::endl;
        cleanup();
        exit(1);
    }

    std::cout<<"Done"<<std::endl;



    csrMatrix csr(nnz, height, csrRowPtr, colIndexDevPtr, valDevPtr);
    return csr;
}

// Private 

void cooMatrix::updateBuffers() {
    if(mstatus == DeviceOutdated) {
        std::cout<<"Copying to device"<<std::endl;
        // copy from host to device
        cudaError_t rowCopy = cudaMemcpy(rowIndexDevPtr, rowIndexHostPtr, 
                                      (size_t)(nnz*sizeof(rowIndexHostPtr[0])),
                                      cudaMemcpyHostToDevice);
        cudaError_t colCopy = cudaMemcpy(colIndexDevPtr, colIndexHostPtr, 
                                      (size_t)(nnz*sizeof(colIndexHostPtr[0])),
                                      cudaMemcpyHostToDevice);
        cudaError_t valCopy = cudaMemcpy(valDevPtr, valHostPtr, 
                                      (size_t)(nnz*sizeof(valHostPtr[0])),
                                      cudaMemcpyHostToDevice);
        if((rowCopy != cudaSuccess) || 
           (colCopy != cudaSuccess) || 
           (valCopy != cudaSuccess)) {
            std::cout<<"Host to device copy failed at line "<<__LINE__<<" in "<<__FILE__<<std::endl;
            cleanup();
        }   
    }else if(mstatus == HostOutdated) {
        std::cout<<"Copying to host"<<std::endl;
        // copy from device to host
        cudaError_t rowCopy = cudaMemcpy(rowIndexDevPtr, rowIndexHostPtr, 
                                      (size_t)(nnz*sizeof(rowIndexHostPtr[0])),
                                      cudaMemcpyDeviceToHost);
        cudaError_t colCopy = cudaMemcpy(colIndexDevPtr, colIndexHostPtr, 
                                      (size_t)(nnz*sizeof(colIndexHostPtr[0])),
                                      cudaMemcpyDeviceToHost);
        cudaError_t valCopy = cudaMemcpy(valDevPtr, valHostPtr, 
                                      (size_t)(nnz*sizeof(valHostPtr[0])),
                                      cudaMemcpyDeviceToHost);
        if((rowCopy != cudaSuccess) || 
           (colCopy != cudaSuccess) || 
           (valCopy != cudaSuccess)) {
            std::cout<<"Device to host copy failed at line "<<__LINE__<<" in "<<__FILE__<<std::endl;
            cleanup();
        }   
    }   
}

void cooMatrix::cleanup() {
    // do something to make sure we can't use them later!
    // free host pointers
    delete[] colIndexHostPtr;
    delete[] rowIndexHostPtr;
    delete[] valHostPtr;
    // free device pointers
    if(rowIndexDevPtr) cudaFree(rowIndexDevPtr);
    if(colIndexDevPtr) cudaFree(colIndexDevPtr);
    if(valDevPtr) cudaFree(valDevPtr);
}

// =============================================================================
// ============================== Dense vector =================================
// =============================================================================

// public

denseVector::denseVector(int _n){
    n = _n;
    // alloc on the host and device
    hostPtr = new float[n];
    cudaError_t alloc_s = cudaMalloc((void**)&devPtr, n*sizeof(float));
    // check for errors
    if((alloc_s != cudaSuccess)) {
        std::cout<<"Device malloc failed at line "<<__LINE__<<" in "<<__FILE__<<std::endl;
        cleanup();
    }
    // assume we start with the device buffers outdated
    mstatus = DeviceOutdated;
    ixPtr = 0;
}

denseVector::~denseVector(){
    cleanup();
}

void denseVector::push(float value) {
    if(!hostPtr){
        std::cout<<"Error: attemping to push elements before allocating."<<std::endl;
    }
    if(ixPtr >= n){
        std::cout<<"Error: allocating too many elements!"<<std::endl;
    }
    hostPtr[ixPtr] = value;
    ixPtr++;
}

void denseVector::print(){
    std::cout<<"[";
    for(int i = 0;i<n;i++){
        if(i == 0){
            std::cout<<hostPtr[i];
        }else{
            std::cout<<","<<hostPtr[i];
        }
    }
    std::cout<<"]"<<std::endl;
}

// private

void denseVector::cleanup(){
    // do something to make sure we can't use them later!
    // free host pointers
    delete[] hostPtr;
    // free device pointers
    if(devPtr) cudaFree(devPtr);
}

#include "mvstructs.h"

// =============================================================================
// =============================== CSR Matrix ==================================
// =============================================================================


csrMatrix::csrMatrix(int _nnz, int _h, int _w, int* _rp, int* _cixp, float* _vp) {
    rowDevPtr = _rp;
    colIxDevPtr = _cixp;
    valDevPtr = _vp;
    nnz = _nnz;
    h = _h;
    w = _w;
}

csrMatrix::~csrMatrix() {
    if(rowDevPtr) cudaFree(rowDevPtr);
    if(colIxDevPtr) cudaFree(colIxDevPtr);
    if(valDevPtr) cudaFree(valDevPtr);
}

void csrMatrix::spmv(cusparseHandle_t handle, denseVector& _x, denseVector& _r){
    // create a matrix descriptor
    cusparseMatDescr_t descr=0;
    cusparseStatus_t status = cusparseCreateMatDescr(&descr);
    if(status != CUSPARSE_STATUS_SUCCESS) {
        std::cout<<"Matrix desriptor initialisation failed"<<std::endl;
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(handle);
        exit(1);
    }
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    // create cuda events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // build a vector y
    denseVector _y = denseVector(w);
    _y.fill(0.0f);

    std::cout<<"_y before: "<<std::endl;
    _y.print();

    std::cout<<"Starting spmv..."<<std::endl;
    // get a pointer to the vector's device memory
    float* x = _x.getDevPtr();
    float* y = _y.getDevPtr();

    float alpha = 1.0;
    float beta = 1.0;

    // perform the csrspmv computation, and time it
    cudaEventRecord(start);
    status= cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, h, w, nnz, &alpha,
        descr, valDevPtr, rowDevPtr, colIxDevPtr, x, &beta, y);
    cudaEventRecord(stop);

    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cout<<"Spmv failed!"<<std::endl;
        cusparseDestroy(handle);
        cusparseDestroyMatDescr(descr);
        exit(3);
    }    

    _y.download();
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<"Took: "<<milliseconds<<"ms"<<std::endl;


    std::cout<<"_y after: "<<std::endl;
    _y.print();


    // copy y into _r;
    _r = _y;

    cusparseDestroyMatDescr(descr);
}


// =============================================================================
// =============================== COO Matrix ==================================
// =============================================================================

// public

cooMatrix::cooMatrix(int _nnz, int _h, int _w) {
    nnz = _nnz;
    h = _h;
    w = _w;
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
        std::cout<<"Device malloc failed before line "<<__LINE__<<" in "<<__FILE__<<std::endl;
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
    // todo: add check that element is within bounds
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
    std::cout<<"Allocating csr memory"<<std::endl;

    cudaError_t alloc_s = cudaMalloc((void**)&csrRowPtr,(h+1)*sizeof(int));
    if(alloc_s != cudaSuccess) {
        std::cout<<"Device malloc failed before line "<<__LINE__<<" in "<<__FILE__<<std::endl;

        cleanup();
        cudaFree(csrRowPtr);
        exit(1);
    }
    std::cout<<"Converting"<<std::endl;

    cusparseStatus_t convert_s = cusparseXcoo2csr(handle, rowIndexDevPtr, nnz, h,
                                           csrRowPtr, CUSPARSE_INDEX_BASE_ZERO);
    if(convert_s != CUSPARSE_STATUS_SUCCESS){
        std::cout<<"Conversion from COO to CSR format failed"<<std::endl;
        cleanup();
        exit(1);
    }

    csrMatrix csr(nnz, h, w, csrRowPtr, colIndexDevPtr, valDevPtr);
    return csr;
}

// Private 

void cooMatrix::updateBuffers() {
    if(mstatus == DeviceOutdated) {
        std::cout<<"Copying matrix to device"<<std::endl;
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
            std::cout<<"Host to device copy failed before line "<<__LINE__<<" in "<<__FILE__<<std::endl;
            cleanup();
        }   
    }else if(mstatus == HostOutdated) {
        std::cout<<"Copying matrix to host"<<std::endl;
        // copy from device to host
        cudaError_t rowCopy = cudaMemcpy(rowIndexHostPtr, rowIndexDevPtr, 
                                      (size_t)(nnz*sizeof(rowIndexHostPtr[0])),
                                      cudaMemcpyDeviceToHost);
        cudaError_t colCopy = cudaMemcpy(colIndexHostPtr, colIndexDevPtr, 
                                      (size_t)(nnz*sizeof(colIndexHostPtr[0])),
                                      cudaMemcpyDeviceToHost);
        cudaError_t valCopy = cudaMemcpy(valHostPtr, valDevPtr, 
                                      (size_t)(nnz*sizeof(valHostPtr[0])),
                                      cudaMemcpyDeviceToHost);
        if((rowCopy != cudaSuccess) || 
           (colCopy != cudaSuccess) || 
           (valCopy != cudaSuccess)) {
            std::cout<<"Device to host copy failed before line "<<__LINE__<<" in "<<__FILE__<<std::endl;
            cleanup();
        }   
    } 
    mstatus = UpToDate;
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
        std::cout<<"Device malloc failed before line "<<__LINE__<<" in "<<__FILE__<<std::endl;
        cleanup();
    }
    // assume we start with the device buffers outdated
    mstatus = DeviceOutdated;
    ixPtr = 0;
}

denseVector::~denseVector(){
    cleanup();
}

denseVector & denseVector::operator= (const denseVector& other) {
    if(this != &other) 
    {
        // delete our memory first, then reallocate
        cleanup();
        n = other.n;
        ixPtr = 0;
        mstatus = other.mstatus;
        // allocate memory
        hostPtr = new float[n];
        cudaError_t alloc_s = cudaMalloc((void**)&devPtr, n*sizeof(float));
        // check for errors
        if((alloc_s != cudaSuccess)) {
            std::cout<<"Device malloc failed before line "<<__LINE__<<" in "<<__FILE__<<std::endl;
            cleanup();
        }
        // copy over
        std::copy(other.hostPtr, other.hostPtr + other.n, hostPtr);
        cudaError_t mcopy = cudaMemcpy(devPtr, other.devPtr,
                                    (size_t)(other.n*sizeof(other.hostPtr[0])),
                                    cudaMemcpyDeviceToDevice);
        if((mcopy != cudaSuccess)) {
            std::cout<<"Device to host copy failed before line "<<
                __LINE__<<" in "<<__FILE__<<std::endl;
            cleanup();
            exit(1);
        }   
    }
    return *this;
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

void denseVector::fill(float value) {
    if(!hostPtr){
        std::cout<<"Error: attemping to push elements before allocating."<<std::endl;
    }
    if(ixPtr >= n){
        std::cout<<"Error: allocating too many elements!"<<std::endl;
    }
    // don't need to update host buffer - we're just about to overwrite it
    for(int i = 0;i<n;i++){
        hostPtr[i] = value;
    }
    mstatus = DeviceOutdated;
    updateBuffers();
}

void denseVector::print(){
    updateBuffers();
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

float* denseVector::getDevPtr(){
    return devPtr;
}

void denseVector::download() {
    std::cout<<"Downloading from the device"<<std::endl;
    mstatus = HostOutdated;
    updateBuffers();
}

// private

void denseVector::cleanup(){
    // do something to make sure we can't use them later!
    // free host pointers
    if(hostPtr){
        delete[] hostPtr;
        hostPtr = 0;
    }
    // free device pointers
    if(devPtr) {
        cudaFree(devPtr);
        devPtr = 0;
    }
}

void denseVector::updateBuffers() {
    if(mstatus == DeviceOutdated) {
        std::cout<<"Copying vector to device"<<std::endl;
        // copy from host to device
        cudaError_t mcopy = cudaMemcpy(devPtr, hostPtr, 
                                      (size_t)(n*sizeof(hostPtr[0])),
                                      cudaMemcpyHostToDevice);
        if((mcopy != cudaSuccess)) {
            std::cout<<"Host to device copy failed before line "<<
                __LINE__<<" in "<<__FILE__<<std::endl;
            cleanup();
        }   
    }else if(mstatus == HostOutdated) {
        std::cout<<"Copying vector to host"<<std::endl;
        // copy from device to host
        cudaError_t mcopy = cudaMemcpy(hostPtr, devPtr, 
                                      (size_t)(n*sizeof(hostPtr[0])),
                                      cudaMemcpyDeviceToHost);
        if((mcopy != cudaSuccess)) {
            std::cout<<"Device to host copy failed before line "<<
                __LINE__<<" in "<<__FILE__<<std::endl;
            cleanup();
        }   
    }
    mstatus = UpToDate;
}

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

std::vector<float> csrMatrix::spmv(cusparseHandle_t handle, denseVector& _x, denseVector& _r){
    // create a matrix descriptor
    cusparseMatDescr_t descr=0;
    cusparseStatus_t status = cusparseCreateMatDescr(&descr);
    if(status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr<<"Matrix desriptor initialisation failed"<<std::endl;
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
    

    std::cerr<<"_y before: "<<std::endl;
    // _y.print();

    std::cerr<<"Starting spmv..."<<std::endl;
    // get a pointer to the vector's device memory
    float* x = _x.getDevPtr();
    float* y = _y.getDevPtr();

    float alpha = 1.0;
    float beta = 1.0;

    int iterations = 30;

    float* runtimes = new float[30];
    for(int i = 0;i<30;i++){
        _y.fill(0.0f);
        // perform the csrspmv computation, and time it
        cudaEventRecord(start);
        status= cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, h, w, nnz, &alpha,
            descr, valDevPtr, rowDevPtr, colIxDevPtr, x, &beta, y);
        cudaEventRecord(stop);

        if (status != CUSPARSE_STATUS_SUCCESS) {
            std::cerr<<"Spmv failed!"<<std::endl;
            cusparseDestroy(handle);
            cusparseDestroyMatDescr(descr);
            exit(3);
        }    

        _y.download();
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        // std::cout<<"Took: "<<milliseconds<<"ms"<<std::endl;
        runtimes[i] = milliseconds;
    }


    std::cerr<<"_y after: "<<std::endl;
    // _y.print();

    // copy y into _r;
    _r = _y;

    cusparseDestroyMatDescr(descr);

    auto rv =  std::vector<float>(runtimes, runtimes + iterations);
    delete[] runtimes;
    return rv;
}


// =============================================================================
// =============================== COO Matrix ==================================
// =============================================================================

// public

cooMatrix::cooMatrix(int _nnz, int _h, int _w) {
    nnz = _nnz;
    h = _h;
    w = _w;
    alloc();
    // set the indexing pointer for adding nonzero elements
    ixPtr = 0;
    // assume we start with the device buffers outdated
    mstatus = DeviceOutdated;
}

cooMatrix::cooMatrix(std::string filename) {
    // parse the file!
    // we have to use crappy C file IO functions,
    // as we're using the mmio libraries
    int ret_code;
    MM_typecode matcode;
    FILE* f;

    if((f = fopen(filename.c_str(), "r")) == NULL) {
        std::cerr<<"Failed to open matrix file!"<<std::endl;
        exit(3);
    }
    // read in the banner
    if(mm_read_banner(f, &matcode) != 0) {
        std::cerr<<"Could not red matrix market banner!"<<std::endl;
        exit(3);
    }
    // check the banner properties
    if(mm_is_matrix(matcode) && 
       mm_is_coordinate(matcode) &&
       (mm_is_real(matcode) || mm_is_integer(matcode) || mm_is_pattern(matcode)) &&
       (mm_is_general(matcode) || mm_is_symmetric(matcode))
    ){
        std::cerr<<"Found matrix of type: "<<matcode<<std::endl;
    }else{
        std::cerr<<"Cannot process matrix of type: "<<matcode<<std::endl;
    }

    // declare matrix size information:
    int _h = -1;
    int _w = -1;
    int _nnz = -1;

    if((ret_code = mm_read_mtx_crd_size(f, &_h, &_w, &_nnz)) != 0){
        std::cerr<<"Cannot read matrix size/number of elements!"<<std::endl;
        exit(3);
    }else{
        std::cerr<<"rows: "<<_h<<" cols: "<<_w<<" nnz: "<<_nnz<<std::endl;
    }

    // a structure to hold triples
    typedef struct coord {
        int row, col;
        float val;
    } coord;
    // vector of values
    std::vector<coord> nzelems;
    int pat = mm_is_pattern(matcode);
    int sym = mm_is_symmetric(matcode);

    // the mmio routines have read up until the first entry, so take advantage of it
    for(int i = 0; i < _nnz; i++){
        coord tmp = {-1, -1, -1.0f};
        if(pat) {
            fscanf(f, "%d %d\n", &(tmp.row), &(tmp.col));
            tmp.val = 1.0;
        }else{
            fscanf(f, "%d %d %f\n", &(tmp.row), &(tmp.col), &(tmp.val));
        }
        std::cerr<<"row: " << tmp.row << " col: " << tmp.col  << " val: " << tmp.val << std::endl;
        tmp.row -= 1;
        tmp.col -= 1;
        nzelems.push_back(tmp);
        // add an element if symmetric - but not repeating the diagonal
        if(sym && (tmp.row != tmp.col)){
            int t = tmp.row;
            tmp.row = tmp.col; tmp.col=t;
            nzelems.push_back(tmp);
        }
        
    }
    // sort the nonzero elements
    auto comparator = [](coord a, coord b){
        if(a.row < b.row){
            return true;
        } else if(a.row > b.row){
            return false;
        } else {
            if(a.col < b.col){
                return true;
            }else{
                return false;
            }
        }
    };
    std::sort(nzelems.begin(), nzelems.end(), comparator);

    // allocate space in the matrix class
    nnz = nzelems.size();
    h = _h;
    w = _w;
    alloc();

    // add the nonzero elements
    for(int i = 0;i<nnz;i++){
        auto c = nzelems[i];
        rowIndexHostPtr[i] = c.row;
        colIndexHostPtr[i] = c.col;
        valHostPtr[i] = c.val;
    }
    // print it
    // print();
    // set the iterator to 0, and the memstatus to device outdated
    ixPtr = 0;
    // assume we start with the device buffers outdated
    mstatus = DeviceOutdated;
}

cooMatrix::~cooMatrix() {
    cleanup();
}

void cooMatrix::push(int col, int row, float val){
    if(!colIndexHostPtr || !rowIndexHostPtr || !valHostPtr){
        std::cerr<<"Error: attemping to push elements before allocating."<<std::endl;
    }
    if(ixPtr >= nnz){
        std::cerr<<"Error: allocating too many elements!"<<std::endl;
        std::cerr<<"ixPtr: " << ixPtr << " >= "<< nnz << " (nnz) "<< std::endl;
    }
    // todo: add check that element is within bounds
    rowIndexHostPtr[ixPtr] = row;
    colIndexHostPtr[ixPtr] = col;
    valHostPtr[ixPtr] = val;
    ixPtr++;
}

void cooMatrix::print() {
    for (int i=0; i<nnz; i++){       
        std::cerr<<"rowIndexHostPtr["<<i<<"] = "<< rowIndexHostPtr[i]<<"  ";
        std::cerr<<"colIndexHostPtr["<<i<<"] = "<< colIndexHostPtr[i]<<"  ";
        std::cerr<<"valHostPtr["<<i<<"] = "<<valHostPtr[i]<<std::endl;
    }
}

csrMatrix cooMatrix::asCSR(cusparseHandle_t handle) {
    updateBuffers();
    std::cerr<<"Building csrmatrix"<<std::endl;
    // get a 
    int * csrRowPtr = 0;
    std::cerr<<"Allocating csr memory"<<std::endl;

    cudaError_t alloc_s = cudaMalloc((void**)&csrRowPtr,(h+1)*sizeof(int));
    if(alloc_s != cudaSuccess) {
        std::cerr<<"Device malloc failed before line "<<__LINE__<<" in "<<__FILE__<<std::endl;

        cleanup();
        cudaFree(csrRowPtr);
        exit(1);
    }
    std::cerr<<"Converting"<<std::endl;

    cusparseStatus_t convert_s = cusparseXcoo2csr(handle, rowIndexDevPtr, nnz, h,
                                           csrRowPtr, CUSPARSE_INDEX_BASE_ZERO);
    if(convert_s != CUSPARSE_STATUS_SUCCESS){
        std::cerr<<"Conversion from COO to CSR format failed"<<std::endl;
        cleanup();
        exit(1);
    }

    csrMatrix csr(nnz, h, w, csrRowPtr, colIndexDevPtr, valDevPtr);
    return csr;
}

// Private 

void cooMatrix::updateBuffers() {
    if(mstatus == DeviceOutdated) {
        std::cerr<<"Copying matrix to device"<<std::endl;
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
            std::cerr<<"Host to device copy failed before line "<<__LINE__<<" in "<<__FILE__<<std::endl;
            cleanup();
        }   
    }else if(mstatus == HostOutdated) {
        std::cerr<<"Copying matrix to host"<<std::endl;
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
            std::cerr<<"Device to host copy failed before line "<<__LINE__<<" in "<<__FILE__<<std::endl;
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

void cooMatrix::alloc() {
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
        std::cerr<<"Device malloc failed before line "<<__LINE__<<" in "<<__FILE__<<std::endl;
        cleanup();
    }
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
        std::cerr<<"Device malloc failed before line "<<__LINE__<<" in "<<__FILE__<<std::endl;
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
            std::cerr<<"Device malloc failed before line "<<__LINE__<<" in "<<__FILE__<<std::endl;
            cleanup();
        }
        // copy over
        std::copy(other.hostPtr, other.hostPtr + other.n, hostPtr);
        cudaError_t mcopy = cudaMemcpy(devPtr, other.devPtr,
                                    (size_t)(other.n*sizeof(other.hostPtr[0])),
                                    cudaMemcpyDeviceToDevice);
        if((mcopy != cudaSuccess)) {
            std::cerr<<"Device to host copy failed before line "<<
                __LINE__<<" in "<<__FILE__<<std::endl;
            cleanup();
            exit(1);
        }   
    }
    return *this;
}

void denseVector::push(float value) {
    if(!hostPtr){
        std::cerr<<"Error: attemping to push elements before allocating."<<std::endl;
    }
    if(ixPtr >= n){
        std::cerr<<"Error: allocating too many elements!"<<std::endl;
    }
    hostPtr[ixPtr] = value;
    ixPtr++;
    mstatus = DeviceOutdated;
}

void denseVector::fill(float value) {
    if(!hostPtr){
        std::cerr<<"Error: attemping to push elements before allocating."<<std::endl;
    }
    // don't need to update host buffer - we're just about to overwrite it
    for(int i = 0;i<n;i++){
        hostPtr[i] = value;
    }
    mstatus = DeviceOutdated;
    updateBuffers();
}

void denseVector::fillRandom() {
    if(!hostPtr){
        std::cerr<<"Error: attemping to push elements before allocating."<<std::endl;
    }
      // c++ random number generation!
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1000.0);
    auto rgen = std::bind(distribution, generator);

    // build the vector
    for(int i = 0;i<n;i++){
        hostPtr[i] = rgen();
    }
    mstatus = DeviceOutdated;
    updateBuffers();
}

void denseVector::print(){
    updateBuffers();
    std::cerr<<"[";
    for(int i = 0;i<n;i++){
        if(i == 0){
            std::cerr<<hostPtr[i];
        }else{
            std::cerr<<","<<hostPtr[i];
        }
    }
    std::cerr<<"]"<<std::endl;
}

float* denseVector::getDevPtr(){
    updateBuffers();
    return devPtr;
}

void denseVector::download() {
    std::cerr<<"Downloading from the device"<<std::endl;
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
        std::cerr<<"Copying vector to device"<<std::endl;
        // copy from host to device
        cudaError_t mcopy = cudaMemcpy(devPtr, hostPtr, 
                                      (size_t)(n*sizeof(hostPtr[0])),
                                      cudaMemcpyHostToDevice);
        if((mcopy != cudaSuccess)) {
            std::cerr<<"Host to device copy failed before line "<<
                __LINE__<<" in "<<__FILE__<<std::endl;
            cleanup();
        }   
    }else if(mstatus == HostOutdated) {
        std::cerr<<"Copying vector to host"<<std::endl;
        // copy from device to host
        cudaError_t mcopy = cudaMemcpy(hostPtr, devPtr, 
                                      (size_t)(n*sizeof(hostPtr[0])),
                                      cudaMemcpyDeviceToHost);
        if((mcopy != cudaSuccess)) {
            std::cerr<<"Device to host copy failed before line "<<
                __LINE__<<" in "<<__FILE__<<std::endl;
            cleanup();
        }   
    }
    mstatus = UpToDate;
}

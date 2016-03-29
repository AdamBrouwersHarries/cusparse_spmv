#pragma once
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <algorithm>
#include <functional>
#include "mmio.h"
#include <cuda_runtime.h>
#include "cusparse.h"


enum MemStatus {
    HostOutdated,
    DeviceOutdated,
    UpToDate
};

class denseVector
{
public:
    denseVector(int);
    ~denseVector();
    denseVector & operator= (const denseVector&);
    void push(float);
    void fill(float);
    void fillRandom();
    void print();
    float* getDevPtr();
    void download();
private:
    float* hostPtr;
    float* devPtr;
    int n;
    int ixPtr;
    MemStatus mstatus;

    void updateBuffers();
    void cleanup();
};

class csrMatrix
{
public:

    csrMatrix(int, int, int, int*, int*, float*);
    ~csrMatrix();
    std::vector<float> spmv(cusparseHandle_t,denseVector&,denseVector&);
private:
    // device pointers 
    int* rowDevPtr;
    int* colIxDevPtr;
    float* valDevPtr;
    int nnz;
    int h;
    int w;
    // no host pointers - we hopefully won't need them
    // buffer status. This should always be "HostOutdated"
    // MemStatus status;
    // void updateBuffers();
};

class cooMatrix
{
public:
    cooMatrix(int, int, int);
    cooMatrix(std::string);
    ~cooMatrix();
    void push(int, int, float);
    // sorting after pushing values
    void print();
    csrMatrix asCSR(cusparseHandle_t);
    int getWidth() {return w;}

private: 
    int nnz;
    int h;
    int w;
    // host pointers
    int * rowIndexHostPtr;
    int * colIndexHostPtr;
    float * valHostPtr;
    // device pointers
    int * rowIndexDevPtr;
    int * colIndexDevPtr;
    float* valDevPtr;
    int ixPtr;
    // transfer status
    MemStatus mstatus;
    void updateBuffers();
    // cleanup functionality
    void cleanup();
    // memory allocation
    void alloc();
};



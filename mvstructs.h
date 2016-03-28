#pragma once
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include "cusparse.h"

enum MemStatus {
    HostOutdated,
    DeviceOutdated
};

class csrMatrix
{
public:

    csrMatrix(int, int, int*, int*, float*);
    ~csrMatrix();
private:
    // device pointers 
    int* rowDevPtr;
    int* colIxDevPtr;
    float* valDevPtr;
    int nnz;
    // no host pointers - we hopefully won't need them
    // buffer status. This should always be "HostOutdated"
    // MemStatus status;
    // void updateBuffers();
};

class cooMatrix
{
public:
    cooMatrix(int, int);
    ~cooMatrix();
    void push(int, int, float);
    void print();
    csrMatrix asCSR(cusparseHandle_t);

private: 
    int nnz;
    int height;
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
};

class denseVector
{
public:
    denseVector(int);
    ~denseVector();
    void push(float);
    void print();
private:
    float* hostPtr;
    float* devPtr;
    int n;
    int ixPtr;
    MemStatus mstatus;

    void cleanup();
};

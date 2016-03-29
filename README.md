# cusparse spmv

A small example program for benchmarking cuSPARSE's csrmv routine with real-world
data, against a randomly initialised vector.

Calling `make` should be sufficient to build the example program.
The example can then be called as follows: `./bin/csrspmv <matrixfile> <matrixname> <hostname>`
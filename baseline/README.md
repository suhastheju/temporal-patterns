# Baseline

## Overview
This software repository contains an experimental software implementation of algorithms for solving a set of pattern-detection problems in temporal graphs. The software is written in C programming language.

This version of the source code is realeased for SDM - 2020 submission titled "Pattern detection in large temporal graphs using algebraic fingerprints".

## License
The source code is subject to MIT license.

## Compilation
The source code is configured for a gcc build. Other builds are possible but it might require manual configuration of the 'Makefile'.

Use GNU make to build the software. Check 'Makefile' for more details.

`make clean all`

## Using the software
Usage: 

`./BASE_PAR <-first> -pre <0/1> -in <input file> -seed <random seed>`

Arguments:

        -first       		: <first> extract a solution
                             
        -pre <0/1/2/3>      : <0> no preprocessing
                              <1> preprocessing step-1
        -in <input file>    : read from <input file>
                              read from <stdin> by default
        -seed <random seed> : random seed input

## Input file format
We use dimacs format for the input graph. An example of input graph is available in `input-graph.g`. See `graph-gen` for graph generator. 

## Example

`$ ./BASE_PAR -ascii -pre 0 -dfs -in input-graph.g`  

        invoked as: ./BASE_PAR -ascii -pre 0 -dfs -in input-graph.g
        no random seed given, defaulting to 123456789
        random seed = 123456789
        no max iterations give, defaulting to 10000
        max iterations = 10000
        input: n = 100, m = 1040, k = 5, t = 100 [0.72 ms] {peak: 0.00GiB} {curr: 0.00GiB}
        build query: [zero: 0.23 ms] [pos: 0.09 ms] [adj: 0.10 ms] [adjsort: 0.04 ms] [shade: 0.06 ms] done. [0.56 ms] {peak: 0.00GiB} {curr: 0.00GiB}
        no preprocessing, default execution
        command: baseline dfs
        solution [11, 101.38ms]: [39, 58, 2] [58, 17, 6] [17, 69, 10] [69, 84, 11]
        baseline [dfs]: [init: 0.38 ms] [dfs: 101.38 ms] done. [101.78 ms] -- true
        command done [0.00 ms 101.79 ms 101.79 ms 101.79 ms]
        grand total [103.22 ms] {peak: 0.00GiB}
        host: maagha
        build: multithreaded
        compiler: gcc 9.1.0

The line `command done [0.00 ms 101.79 ms 101.79 ms 101.79 ms]` specifies the runtime of execution. Here the first time `0.00 ms` is preprocessing time, command time is `101.79 ms` and total time is `101.79 ms`.  

The reported runtimes are in milliseconds.

The line `solution [12, 101.38ms]: [39, 58, 2] [58, 17, 6] [17, 69, 10] [69, 84, 11]` reports a solution and each term is of the form `[u, v, i]` where `u, v` are vertices and `i` is the timestamp of transition.

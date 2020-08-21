# gfp8
Search for GFP-*8*

## About

**gfp8** is a multithreaded C++ application.  
It searches for *b* such that *b*<sup>2<sup>*k*</sup></sup> + 1 are primes for 0 &le; *k* &le; 7.
Any number *b* < 2<sup>95</sup> can be tested.  
The integer sequence is https://oeis.org/Axxxxxx.

## Build

gfp8 must be compiled with gcc on Windows or Linux and linked with the GNU Multiple Precision Arithmetic library (GMP) and OpenMP.  

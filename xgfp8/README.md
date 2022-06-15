# xgfp8
Search for xGFP-*8*

## About

**xgfp8** is a multithreaded C++ application.  
It searches for *a*, *b* such that *a*<sup>2<sup>*k*</sup></sup> + *b*<sup>2<sup>*k*</sup></sup> are primes for 0 &le; *k* &le; 7.
Any number *a*, *b* < 2<sup>61</sup> can be tested.  
The integer sequence is https://oeis.org/A343121.

## Build

xgfp8 must be compiled with gcc on Windows or Linux and linked with the GNU Multiple Precision Arithmetic library (GMP).  

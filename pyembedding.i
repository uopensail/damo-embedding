%module pyEmbedding

%{
#define SWIG_FILE_WITH_INIT
#include "include/pyembedding.h"

%}

%include "numpy.i"
%include "std_string.i"
%init
%{
import_array();
%}

%apply (float* IN_ARRAY1, int DIM1) {(float* gds, int gn)}
%apply (unsigned long long* IN_ARRAY1, int DIM1) {(unsigned long long *keys, int kn)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* w, int wn)}

%include "include/pyembedding.h"

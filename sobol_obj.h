#ifndef INCLUDED_SOBOL_OBJ
#define INCLUDED_SOBOL_OBJ

#include <Python.h>
#include "structmember.h"
#define NO_IMPORT_ARRAY
#include <pyarray.h>
#include <gsl/gsl_qrng.h>


typedef struct {
    PyObject_HEAD
    gsl_qrng* _rng;
    int       _size;
    int       _dims;
} SobolSampler;

SobolSampler* SobolSampler_New(size_t size, int dims);

int SobolSampler_Register(PyObject* m);


#endif /* INCLUDED_SOBOL_OBJ */

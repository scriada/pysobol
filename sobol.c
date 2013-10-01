#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <gsl/gsl_qrng.h>

static const char sobol__doc__[] =
    "sobol(size, dims)\n"
    "Sample from a Sobol sequence, where dims is the number of dimensions and size is the number of samples.\n"
    "Returns a numpy.ndarray with dimensions (size, dims).";
static PyObject* sobol(PyObject *self, PyObject *args, PyObject* keywds)
{
    int num_samples = 0;
    int dims        = 0;

    static char *kwlist[] = {"size", "dims", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "ii", kwlist, &num_samples, &dims))
        return NULL;

    gsl_qrng* rng = gsl_qrng_alloc(gsl_qrng_sobol, dims);

    if (rng == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "error occurred");
        return NULL;
    }

    int arr_dims[2]    = {num_samples, dims};
    int nd             = dims == 1 ? 1 : 2; // when dims=1 make arr a vector
    PyArrayObject* arr = (PyArrayObject *)PyArray_SimpleNew(nd, arr_dims, NPY_DOUBLE);
    double* buf        = (double *)PyArray_DATA(arr);

    // TODO - check arr is ok

    for (int i=0; i<num_samples; ++i)
        gsl_qrng_get(rng, (buf + dims*i));

    gsl_qrng_free(rng);

    return PyArray_Return(arr);
}

// method declaration list
static PyMethodDef methods[] = {
    {"sobol", (PyCFunction)sobol, METH_VARARGS|METH_KEYWORDS, sobol__doc__},
    {NULL, NULL, 0, NULL} // Sentinel
};

PyMODINIT_FUNC initsobol(void)
{
    PyObject* m = Py_InitModule3("sobol", methods, "SOBOL sequence generator");

    if (m == NULL)
        return;

    import_array(); // setup numpy
}

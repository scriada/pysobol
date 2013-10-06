#include <sobol_obj.h>
#include <limits.h>


static void
SobolSampler_dealloc(SobolSampler* self)
{
    gsl_qrng_free(self->_rng);
    self->ob_type->tp_free((PyObject*)self);
}


static const char SobolSampler_dims__doc__[] =
    "Number of dimensions";
static PyObject*
SobolSampler_dims(PyObject* self)
{
    SobolSampler* s = (SobolSampler *)self;
    return PyInt_FromSsize_t(s->_dims);
}

static const char SobolSampler_size__doc__[] =
    "Size of the sampler";
static PyObject*
SobolSampler_size(PyObject* self)
{
    SobolSampler* s = (SobolSampler *)self;

    if (s->_is_inf)
    {
        Py_INCREF(Py_None);
        return Py_None;
    }
    else
        return PyInt_FromSsize_t(s->_size);
}

static PyObject*
SobolSampler_iter(PyObject *self)
{
    Py_INCREF(self);
    return self;
}

static PyObject*
SobolSampler_iternext(PyObject *self)
{
    SobolSampler* s = (SobolSampler *)self;

    if (s->_size == 0)
    {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }
    
    int arr_dims[1]    = {s->_dims};
    PyArrayObject* arr = (PyArrayObject *)PyArray_SimpleNew(1 /*nd*/, arr_dims, NPY_DOUBLE);

    if (!arr) return NULL;

    gsl_qrng_get(s->_rng, (double *)PyArray_DATA(arr));
    if (!s->_is_inf) --s->_size;

    return PyArray_Return(arr);
}

static PyMemberDef SobolSampler_members[] = {
    {NULL}  /* Sentinel */
};

static PyMethodDef SobolSampler_methods[] = {
    {"dims", (PyCFunction)SobolSampler_dims, METH_NOARGS, SobolSampler_dims__doc__},
    {"size", (PyCFunction)SobolSampler_size, METH_NOARGS, SobolSampler_size__doc__},
    {NULL}  /* Sentinel */
};

static PyTypeObject SobolSamplerType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "sobol._SoolSampler",      /*tp_name*/
    sizeof(SobolSampler),      /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)SobolSampler_dealloc,      /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_ITER, /*tp_flags*/
    "Sobol sampler",           /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    SobolSampler_iter,         /* tp_iter */
    SobolSampler_iternext,     /* tp_iternext */
    SobolSampler_methods,      /* tp_methods */
    SobolSampler_members,      /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    0,                         /* tp_new */
};

int SobolSampler_Register(PyObject* m)
{
    SobolSamplerType.tp_new = PyType_GenericNew;

    int ret = 0;
    if ((ret = PyType_Ready(&SobolSamplerType)) < 0)
        return ret;

    Py_INCREF(&SobolSamplerType);
    PyModule_AddObject(m, "_SobolSampler", (PyObject *)&SobolSamplerType);

    return 0;
}

SobolSampler* SobolSampler_New(size_t dims, PyObject* size)
{
    size_t _size   = 0;
    size_t _is_inf = 0;

    if (size == Py_None)
    {
        _size   = 1;
        _is_inf = 1;
    }
    else if (PyInt_Check(size) || PyLong_Check(size))
    {
        _size   = PyInt_AsSsize_t(size);
        _is_inf = 0;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "size must be an integer, or None");
        return NULL;
    }
    
    SobolSampler* s = PyObject_New(SobolSampler, &SobolSamplerType);
    if (!s) return NULL;

    s->_size   = _size;
    s->_is_inf = _is_inf;
    s->_dims   = dims;
    s->_rng    = gsl_qrng_alloc(gsl_qrng_sobol, dims);

    return s;
}

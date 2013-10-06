#include <sobol_obj.h>

static void
SobolSampler_dealloc(SobolSampler* self)
{
    gsl_qrng_free(self->_rng);
    self->ob_type->tp_free((PyObject*)self);
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

    if (arr == NULL)
        return NULL;

    gsl_qrng_get(s->_rng, (double *)PyArray_DATA(arr));
    --s->_size;

    return PyArray_Return(arr);
}

static PyMemberDef SobolSampler_members[] = {
    {NULL}  /* Sentinel */
};

static PyMethodDef SobolSampler_methods[] = {
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
    if (PyType_Ready(&SobolSamplerType) < 0)
        return -1;

    Py_INCREF(&SobolSamplerType);
    PyModule_AddObject(m, "_SobolSampler", (PyObject *)&SobolSamplerType);

    return 0;
}

SobolSampler* SobolSampler_New(size_t size, int dims)
{
    SobolSampler* s = PyObject_New(SobolSampler, &SobolSamplerType);

    if (s == NULL)
        return NULL;

    s->_size = size;
    s->_dims = dims;
    s->_rng = gsl_qrng_alloc(gsl_qrng_sobol, dims);

    return s;
}

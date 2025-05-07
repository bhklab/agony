#include <nanobind/nanobind.h>

int add(int a, int b) { return a + b; } 

NB_MODULE(agony_project, m)
{
    m.def("add", &add); 
}
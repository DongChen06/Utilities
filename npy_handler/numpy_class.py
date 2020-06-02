# https://stackoverflow.com/questions/15454285/numpy-array-of-class-instances/15455053
import numpy

class Atom(object):
    def atoms_method(self, foo, bar):
        #...with foo and bar being arrays of Paramsof length m & n
        atom_out = foo + bar
        return atom_out


array = numpy.ndarray((10,),dtype=numpy.object)

for i in xrange(10):
    array[i] = Atom()

for i in xrange(10):
    print array[i].atoms_method(i, 5)

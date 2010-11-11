try:
    import Cython
    has_cython = True
except ImportError:
    has_cython = False

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy import get_include
    config = Configuration('measurements', parent_package, top_path)

    # if Cython is present, then try to build the pyx source
    if has_cython:
        src = ['_realnoiselet.pyx']
    else:
        src = ['_realnoiselet.c']
    config.add_extension('_realnoiselet', src, include_dirs=[get_include()])
    
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

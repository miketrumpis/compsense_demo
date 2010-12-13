# --yoinked from DIPY, thanks!--

# Simple makefile to quickly access handy build commands for Cython extension
# code generation.  Note that the actual code to produce the extension lives in
# the setup.py file, this Makefile is just meant as a command
# convenience/reminder while doing development.

PKGDIR=csdemo

help:
	@echo "Numpy/Cython tasks.  Available tasks:"
	@echo "ext  -> build the Cython extension module."
	@echo "html -> create annotated HTML from the .pyx sources"
	@echo "all  -> builds ext"

all: ext

ext: realnoiselet.so jpgzzind.so

html:  ${PKGDIR}/measurements/realnoiselet.html ${PKGDIR}/utils/jpgzzind.html

realnoiselet.so: ${PKGDIR}/measurements/realnoiselet.c
jpgzzind.so: ${PKGDIR}/utils/jpgzzind.c

	python setup.py build_ext --inplace

# Phony targets for cleanup and similar uses

.PHONY: clean

clean:
	- find ${PKGDIR} -name "*.so" -print0 | xargs -0 rm
	- find ${PKGDIR} -name "*.c" -print0 | xargs -0 rm
	- find ${PKGDIR} -name "*.html" -print0 | xargs -0 rm
	- find . -name "*.pyc" -print0 | xargs -0 rm
	rm -rf build

# Suffix rules
%.c : %.pyx
	cython $<

%.html : %.pyx
	cython -a $<


#!/bin/sh

export HAS_NETCDF4=1
export HAS_HDFEOS=1
export HAS_HDFEOS5=1
export HAS_GDAL=1
export HAS_GRIB2=1
export NETCDF_PREFIX=${PREFIX}
export F2CLIBS=gfortran
export HAS_SZIP=0
export HAS_HDF4=1
export HAS_HDF5=1
export HAS_GDAL=1

export CXXFLAGS="-fPIC $CXXFLAGS"
export LDFLAGS="-L$PREFIX/lib $LDFLAGS"
export CPPFLAGS="-I$PREFIX/include $CPPFLAGS"
export CFLAGS="-D_BSD_SOURCE -D_XOPEN_SOURCE -I$PREFIX/include $CFLAGS"

if [[ $(uname) == Darwin ]]; then
  export CC=clang
  export CXX=clang++
  export MACOSX_DEPLOYMENT_TARGET="10.9"
  export CXXFLAGS="-stdlib=libc++ $CXXFLAGS"
  export CXXFLAGS="$CXXFLAGS -stdlib=libc++"
  export LDFLAGS="-headerpad_max_install_names $LDFLAGS"
fi

$PYTHON setup.py install

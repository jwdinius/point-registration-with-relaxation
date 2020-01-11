installprefix=/usr/local
srcdir=/Ipopt-3.12.13

echo "Building Ipopt from ${srcdir}"
echo "Saving headers and libraries to ${installprefix}"

# BLAS
cd $srcdir/ThirdParty/Blas
./get.Blas
mkdir -p build && cd build
../configure --prefix=$installprefix --disable-shared --with-pic
make install

# Lapack
cd $srcdir/ThirdParty/Lapack
./get.Lapack
mkdir -p build && cd build
../configure --prefix=$installprefix --disable-shared --with-pic \
    --with-blas="$installprefix/lib/libcoinblas.a -lgfortran"
make install

# ASL
cd $srcdir/ThirdParty/ASL
./get.ASL

# MUMPS
cd $srcdir/ThirdParty/Mumps
./get.Mumps

# build everything
cd $srcdir
./configure --prefix=$installprefix coin_skip_warn_cxxflags=yes \
    --with-blas="$installprefix/lib/libcoinblas.a -lgfortran" \
    --with-lapack=$installprefix/lib/libcoinlapack.a
make
make test
make -j1 install

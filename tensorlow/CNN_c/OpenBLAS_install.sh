git clone git://github.com/xianyi/OpenBLAS
cd OpenBLAS
sudo apt-get install gfortran
make FC=gfortran
make PREFIX=./Openblas install
sudo cp ./Openblas/include/* /usr/local/include/

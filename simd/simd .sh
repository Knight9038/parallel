#simd.sh
#!/bin/sh
#PBS -N simd

pssh -h $PBS_NODEFILE mkdir -p /home/s2211804/SIMD 1>&2
scp master:/home/s2211804/SIMD/simd /home/s2211804/SIMD
pscp -h $PBS_NODEFILE master:/home/s2211804/SIMD/simd /home/s2211804/SIMD 1>&2

/home/s2211804/SIMD/simd

# mpi.sh
# !/ bin / sh
# PBS -N mpi
# PBS -l nodes = 4

pssh -h $PBS_NODEFILE mkdir -p /home/s2211804/MPI 1>&2
scp master:/home/s2211804/MPI/mpi /home/s2211804/MPI
pscp -h $PBS_NODEFILE /home/s2211804/MPI/mpi /home/s2211804/MPI 1>&2

mpiexec -np 4 -machinefile $PBS_NODEFILE /home/s2211804/MPI/mpi

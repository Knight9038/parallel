#p_o.sh
#!/bin/sh
#PBS -N p_o

pssh -h $PBS_NODEFILE mkdir -p /home/s2211804/Pthread_OpenMP 1>&2
scp master:/home/s2211804/Pthread_OpenMP/p_o /home/s2211804/Pthread_OpenMP
pscp -h $PBS_NODEFILE master:/home/s2211804/Pthread_OpenMP/p_o /home/s2211804/Pthread_OpenMP 1>&2

/home/s2211804/Pthread_OpenMP/p_o

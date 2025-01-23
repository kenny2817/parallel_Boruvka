all: run

build: MST_MPI.c
	mpicc -Wall -Wextra -o mst_mpi MST_MPI.c -lm

run: build
	mpirun -np 4 ./mst_mpi

clean: mst_mpi
	rm mst_mpi
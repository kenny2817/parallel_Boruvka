all: build

build: MST_MPI.c rnd_graph_generator.c
	gcc -o rnd_graph_generator rnd_graph_generator.c
	mpicc -Wall -Wextra -o mst_mpi MST_MPI.c -lm

clean: mst_mpi
	rm mst_mpi rnd_graph_generator

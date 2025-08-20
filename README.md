# parallel_Boruvka

parallel implemetation of boruvka algorithm using MPI whenever possible, often abusing it. This was an experiment to learn more about this approach and explore this algorithm.

### Future work

- openmp implementation
- hybrid implementation
- HPC testing

### compile

`make`

### generate graph

`./rnd_graph_generator <file.txt> <E> <V> <W>`

file.txt = name of the file where the graph will be saved

E = number of edges

V = number of vertices

W = max weight of each edge

### run

`mpirun -np <x> ./mst_mpi <file.txt>`

x = number of processes

file.txt = file containing the graph.
the structure should be Line 0: number of vertices (V) followed by the number of edges (E), from Line 1 to E each edge should be raprresented as `vertex vertex weight`

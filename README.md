# parallel_Boruvka

This is the experimentation branch

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

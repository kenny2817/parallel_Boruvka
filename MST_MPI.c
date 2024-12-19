#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "graph.h"

#define MAX_EDGES_FOR_VERTIX(vertix) ((vertix * (vertix - 1)) / 2)
#define FOR(i, start, stop) for (i = start; i < stop; ++i)
#define OLD_G graph[iteration]
#define NEW_G graph[iteration + 1]

#define INPUT "graph.txt"
#define OUTPUT "mst.txt"

typedef int TYP;
#define MPI_TYP MPI_INT

void boruvka(const Graph_CSR *, TYP *, const int, const int);

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, commsz;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsz);

    // converter from other formats
    Graph_CSR G;
    TYP *MST;
    // broadcasting

    boruvka(&G, MST, rank, commsz);
    // reduce mst?

    MPI_Finalize();
    return 0;
}

void compute_partitions(int *, int *, TYP *, TYP *, const int, const int, const int);
void find_min_edge(const Graph_CSR *, TYP *, const int);
void merge_edge(const Graph_CSR *, TYP *, const int *, TYP *, TYP *, const int, int *);
void parent_compression(TYP *, const int);
void super_vertex_init(TYP *, const TYP *, const TYP, const TYP);
void mpi_exclusive_prefix_sum(int *, int *, const TYP, const TYP);
void out_degree_init(int *, const Graph_CSR *, const TYP *, int *, const TYP, const TYP);
void allocate_new_graph(Graph_CSR *, const int *, const int, const int);
void reduce_edges(const Graph_CSR *, Graph_CSR *, const TYP *, const TYP *, const int *, int *, const TYP, const TYP);

// while number of vertices > 1 do
// 2: Find minimum edge per vertex
// 3: Remove mirrored edges
// 4: Initialize colors
// 5: while not converged do
// 6: Propagate colors
// 7: Create new vertex ids
// 8: Count new edges
// 9: Assign edge segments to new vertices
// 10: Insert new edges
void boruvka(const Graph_CSR *G, TYP *MST, const int rank, const int commsz) {
    int iteration, next_iter_v, next_iter_e, internal_index;
    TYP start, stop, partition, missing, i, ii, count, istart, istop;
    MPI_Request req[3];

    Graph_CSR **graph = (Graph_CSR **)malloc(((int)log2f((float)G->V) + 1) * sizeof(Graph_CSR *));
    TYP *min_edge = malloc(G->V * sizeof(TYP));  // if multiple of rank -> less overhead
    TYP *parent = malloc(G->V * sizeof(TYP));
    TYP *super_vertex = calloc(G->V, sizeof(TYP));

    int *recvCounts = malloc(commsz * sizeof(int));
    int *recvCounts1 = malloc(commsz * sizeof(int));
    int *displs = malloc(commsz * sizeof(int));
    int *displs1 = malloc(commsz * sizeof(int));

    int *edge_map = malloc(G->E * sizeof(int));
    int *edge_map_tmp = malloc(G->E * sizeof(int));

    iteration = 0;
    OLD_G = G;

    // tosta...
    start = OLD_G->first_edge[start];
    stop = OLD_G->first_edge[stop] + OLD_G->out_degree[stop];
    FOR(i, start, stop) edge_map_tmp[i] = 1;
    mpi_exclusive_prefix_sum(edge_map_tmp, edge_map, start, stop);

    compute_partitions(recvCounts, displs, &start, &stop, OLD_G->V, rank, commsz);
    FOR(i, start, stop) min_edge[i] = -1;

    while (OLD_G->V > 0) {
        // find min edgess for each v || uneven workload
        FOR(i, start, stop) find_min_edge(OLD_G, min_edge, i);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, min_edge, recvCounts, displs, MPI_TYP, MPI_COMM_WORLD);

        // remove mirrors + merge mfset
        FOR(i, start, stop) merge_edge(OLD_G, MST, edge_map, min_edge, parent, i, &internal_index);
        // here i should save the edges i want to put in mst
        // for now we save partial in mst and final gather will do the trick
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, parent, recvCounts, displs, MPI_TYP, MPI_COMM_WORLD);

        // compression
        // the omp version should be better as there is the possibility o skipping more parents in the process as they are always updated
        // the mpi version has less possibility to skip the more the commsz increase but there is no overhead other that the all gatherv
        // hibrid seem difficult as there would be race conditions
        FOR(i, start, stop) parent_compression(parent, i);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, parent, recvCounts, displs, MPI_TYP, MPI_COMM_WORLD);

        // exclusive prefixsum
        super_vertex_init(super_vertex, parent, start, stop);
        // MPI_Allreduce(MPI_IN_PLACE, super_vertex, vmin_edgeertices, MPI_TYP, MPI_SUM, MPI_COMM_WORLD); // could be better as less complex
        MPI_Reduce_scatter(MPI_IN_PLACE, super_vertex, recvCounts, MPI_TYP, MPI_SUM, MPI_COMM_WORLD);
        next_iter_v = -super_vertex[OLD_G->V - 1];                          // ensure correct size of next iteration
        mpi_exclusive_prefix_sum(super_vertex, super_vertex, start, stop);  // in place
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, super_vertex, recvCounts, displs, MPI_TYP, MPI_COMM_WORLD);

        FOR(i, start, stop) parent[i] = super_vertex[parent[i]];                                                           // improves locality, not sure if needed
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, parent, recvCounts, displs, MPI_TYP, MPI_COMM_WORLD, req[0]);  // by inverting order we could make things better with igatherv

        // early exit here i suppose? --------------
        /// NEW GRAPH
        // out edges
        next_iter_v += super_vertex[OLD_G->V - 1];
        int *new_out_degree = calloc(next_iter_v, sizeof(int));
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);
        out_degree_init(new_out_degree, OLD_G, parent, &next_iter_e, start, stop);
        MPI_Iallreduce(MPI_IN_PLACE, &new_out_degree, next_iter_v, MPI_TYP, MPI_SUM, MPI_COMM_WORLD, &req[0]);
        MPI_Iallreduce(MPI_IN_PLACE, &next_iter_e, 1, MPI_TYP, MPI_SUM, MPI_COMM_WORLD, &req[1]);

        MPI_Waitall(2, req, MPI_STATUS_IGNORE);  // new_out_degree + next_iter_e
        // allocation
        allocate_new_graph(NEW_G, new_out_degree, next_iter_e, next_iter_v);
        compute_partitions(recvCounts, displs, &start, &stop, next_iter_v, rank, commsz);  // new start + stop + recvCounts + displs

        // first edge
        mpi_exclusive_prefix_sum(NEW_G->out_degree, NEW_G->first_edge, start, stop);
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, NEW_G->first_edge, recvCounts, displs, MPI_TYP, MPI_COMM_WORLD, req[0]);

        // edges
        reduce_edges(OLD_G, NEW_G, parent, super_vertex, edge_map, edge_map_tmp, start, stop);
        int start1 = NEW_G->first_edge[start];
        int stop1 = NEW_G->first_edge[stop] + NEW_G->out_degree[stop];
        compute_recvCounts_dipls(recvCounts1, displs1, start1, stop1, rank);
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, NEW_G->destination, recvCounts1, displs1, MPI_TYP, MPI_COMM_WORLD, &req[1]);
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, NEW_G->weight, recvCounts1, displs1, MPI_TYP, MPI_COMM_WORLD, &req[2]);
        MPI_Iallgatherv(edge_map_tmp, stop1 - start1, MPI_INT, edge_map, recvCounts, displs, MPI_INT, MPI_COMM_WORLD, &req[3]);

        MPI_Waitall(3, req, MPI_STATUS_IGNORE);  // first_edge + destination + weight
        MPI_Wait(&req[3], MPI_STATUS_IGNORE);    // edge_map

        iteration++;
    }

    // MST gather
    compute_recvCounts_dipls(recvCounts, displs, 0, internal_index, rank);
    MPI_Igatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, MST, recvCounts, displs, MPI_INT, 0, MPI_COMM_WORLD, &req[0]);

    free(edge_map);
    free(edge_map_tmp);
    for (i = 1; i < iteration; ++i) free_graph(graph[i]);
    free(graph);
    free(min_edge);
    free(parent);
    free(recvCounts);
    free(recvCounts1);
    free(displs);
    free(displs1);

    MPI_Wait(&req[0], MPI_STATUS_IGNORE);
}

void compute_recvCounts_dipls(int *recvCounts, int *displs, const int start, const int stop, const int rank) {
    recvCounts[rank] = stop - start;
    MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvCounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Exscan(recvCounts, displs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

void compute_partitions(int *recvCounts, int *displs, TYP *start, TYP *stop, const int size, const int rank, const int commsz) {
    // partition define as well as buffers for gatherv
    int partition = size / commsz;
    int missing = size % commsz;
    *start = rank * partition;
    *stop = *start + partition;
    if (rank < missing) {
        *start += partition;
    } else {
        *start += missing;
        *stop -= 1;
    }
    compute_recvCounts_dipls(recvCounts, displs, start, stop, rank);
}

void find_min_edge(const Graph_CSR *G, TYP *min_edge, const int i) {
    int first_index = G->first_edge[i];
    int min = first_index;
    for (int j = first_index + 1; j < first_index + G->out_degree[i]; ++j) {
        if (G->weight[j] < G->weight[min]) {
            min = j;
        }
    }
    min_edge[i] = min;
}

void merge_edge(const Graph_CSR *G, TYP *MST, const int *map_edge, TYP *min_edge, TYP *parent, const int i, int *internal_index) {
    int dest = G->destination[i];
    int dest_dest = G->destination[min_edge[dest]];
    if (dest >= 0 && !(dest < i && dest_dest == i)) {  // if init + if mirror only save the one pointing to the smalles one
        parent[i] = dest;                              // mfset init
        MST[*internal_index++] = map_edge[i];          // MST store
    } else {                                           //
        parent[i] = i;                                 // mfset init
    }
    min_edge[i] = -1;  // min_edge reset for next iter
}

void parent_compression(TYP *parent, const int i) {
    int father, grand_father;
    do {
        parent[i] = parent[parent[i]];  // jump liv
        father = parent[i];
        grand_father = parent[father];
    } while (father != grand_father);
}

void super_vertex_init(TYP *super_vertex, const TYP *parent, const TYP start, const TYP stop) {
    int i;
    FOR(i, start, stop)
    if (parent[i] != -1)  // if not connected
        super_vertex[parent[i]] = 1;
}

void mpi_exclusive_prefix_sum(int *send, int *recv, const TYP start, const TYP stop) {
    int i, local_sum = 0, offset = 0, tmp;

    FOR(i, start, stop) {  // Compute local prefix sum
        tmp = send[i];
        recv[i] = local_sum;
        local_sum += tmp;
    }

    MPI_Exscan(&local_sum, &offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);  // get offset

    FOR(i, start + 1, stop) recv[i] += offset;  // apply offset
}

void out_degree_init(int *new_out_degree, const Graph_CSR *Gin, const TYP *parent, int *next_iter_e, const TYP start, const TYP stop) {
    int i, ii, istart, istop;
    FOR(i, start, stop) {
        istart = Gin->first_edge[i];
        istop = istart + Gin->out_degree[i];
        FOR(ii, istart, istop) {
            if (parent[Gin->destination[ii]] != parent[i]) {
                new_out_degree[parent[i]]++;
                next_iter_e++;
            }
        }
    }
}

void allocate_new_graph(Graph_CSR *G, const int *out_degree, const int E, const int V) {
    G = base_graph(E, V);  // we have the correct values
    G->out_degree = out_degree;
    G->first_edge = malloc(V * sizeof(int));
    G->destination = malloc(E * sizeof(int));
    G->weight = malloc(E * sizeof(int));
}

int last_binary_search(const TYP *super_vertex, int size, int lookfor) {
    int left = 0, right = size - 1, mid;
    while (left < right) {
        mid = (left + right) / 2;
        if (super_vertex[mid] <= lookfor)
            left = mid;
        else
            right = mid;
    }
    return right;
}

void reduce_edges(const Graph_CSR *Gin, Graph_CSR *Gout, const TYP *parent, const TYP *super_vertex, const int *edge_map, int *edge_map_tmp, const TYP start, const TYP stop) {
    int i, k, from, to, master_parent;
    int first_vertex_in = last_binary_search(super_vertex, Gin->V, start);  // cheap heuristic 1 // to be revisited ---------------------------

    int edge_out = Gout->first_edge[start];
    int tot_edge = Gout->first_edge[stop] + Gout->out_degree[stop];
    FOR(i, first_vertex_in, Gin->V) {  // for each possible vertex in Gin
        master_parent = parent[i];
        if (master_parent >= start && master_parent < stop) {  // if vertex parent is within range
            from = Gin->first_edge[i];
            to = from + Gin->out_degree[i];
            FOR(k, from, to) {                                          // for edges of that vertex
                if (parent[Gin->destination[k]] != master_parent) {     // if edge cross parent
                    Gout->destination[edge_out] = Gin->destination[k];  // add edge
                    Gout->weight[edge_out] = Gin->weight[k];            //
                    edge_map_tmp[edge_out] = edge_map[k];               // sync mapping
                    edge_out++;                                         // increment counter
                }
            }
            if (edge_out >= tot_edge) break;  // cheap heuristic 2
        }
    }
}

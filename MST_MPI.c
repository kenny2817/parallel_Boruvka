#include <math.h>
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

void boruvka(const Graph_CSR *, const int, const int, TYP *);

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, commsz;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsz);

    int destination[] = {1, 3, 4, 0, 2, 4, 2, 4, 6, 0, 4, 5, 0, 1, 2, 3, 5, 6, 3, 4, 6, 2, 4, 6};
    int weight[] = {4, 12, 9, 4, 8, 7, 8, 5, 2, 12, 3, 6, 9, 7, 5, 3, 1, 13, 6, 1, 11, 2, 13, 11};
    int first_edge[] = {0, 3, 6, 9, 12, 18, 21};
    int out_degree[] = {3, 3, 3, 3, 6, 3, 3};

    // converter from other formats
    Graph_CSR *G = default_graph(12, 7);
    for (int i = 0; i < G->E * 2; ++i) {
        G->destination[i] = destination[i];
        G->weight[i] = weight[i];
    }
    for (int i = 0; i < G->V; ++i) {
        G->first_edge[i] = first_edge[i];
        G->out_degree[i] = out_degree[i];
    }

    TYP *MST = malloc(((G->V - 1)) * sizeof(TYP));

    boruvka(G, rank, commsz, MST);

    if (!rank) {
        int tot;
        for (int i = 0; i < G->V - 1; ++i) {
            tot += weight[MST[i]];
            printf("%d ", MST[i]);
        }
        printf("\ntot= %d", tot);
    }

    free(MST);
    MPI_Finalize();
    return 0;
}

void compute_start_stop_vertices(const Graph_CSR *, const int, const int, TYP *, TYP *);
void compute_start_stop_edges(const Graph_CSR *, const TYP, const TYP, TYP *, TYP *);
void compute_recvCounts_dipls(const int, const int, const int, int *, int *);
void compute_partitions(const Graph_CSR *, const int, const int, TYP *, TYP *, int *, int *);
void find_min_edge(const Graph_CSR *, const int, const int, TYP *);
void merge_edge(const Graph_CSR *, const int *, const int, const int, TYP *, TYP *, TYP *, int *);
void parent_compression(const int, const int, TYP *);
void super_vertex_init(const TYP *, const TYP, const TYP, TYP *);
void mpi_exclusive_prefix_sum(const TYP, const TYP, int *, int *);
void out_degree_init(const Graph_CSR *, const TYP *, const TYP, const TYP, int *, int *);
Graph_CSR *allocate_new_graph(const int *, const int, const int);
void reduce_edges(const Graph_CSR *, const TYP *, const TYP *, const int *, const TYP, const TYP, Graph_CSR *, int *);

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
void boruvka(const Graph_CSR *G, const int rank, const int commsz, TYP *MST) {
    int iteration, next_iter_v, next_iter_e, internal_index;
    TYP start_v, stop_v, start_e, stop_e, i;
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
    compute_start_stop_vertices(OLD_G, rank, commsz, &start_v, &stop_v);
    compute_start_stop_edges(OLD_G, start_v, stop_v, &start_e, &stop_e);
    FOR(i, start_e, stop_e) edge_map_tmp[i] = 1;
    mpi_exclusive_prefix_sum(start_e, stop_e, edge_map_tmp, edge_map);
    compute_recvCounts_dipls(start_e, stop_e, rank, recvCounts, displs);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, edge_map, recvCounts, displs, MPI_INT, MPI_COMM_WORLD);

    FOR(i, start_v, stop_v) min_edge[i] = -1;
    compute_recvCounts_dipls(start_v, stop_v, rank, recvCounts, displs);

    while (OLD_G->V > 1) {
        // find min edgess for each v || uneven workload
        find_min_edge(OLD_G, start_v, stop_v, min_edge);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, min_edge, recvCounts, displs, MPI_TYP, MPI_COMM_WORLD);

        // remove mirrors + merge mfset
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);  // edge_map
        merge_edge(OLD_G, edge_map, start_v, stop_v, MST, min_edge, parent, &internal_index);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, parent, recvCounts, displs, MPI_TYP, MPI_COMM_WORLD);

        // compression
        // the omp version should be better as there is the possibility o skipping more parents in the process as they are always updated
        // the mpi version has less possibility to skip the more the commsz increase but there is no overhead other that the all gatherv
        // hibrid seem difficult as there would be race conditions
        parent_compression(start_v, stop_v, parent);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, parent, recvCounts, displs, MPI_TYP, MPI_COMM_WORLD);

        // supervertex init
        super_vertex_init(parent, start_v, stop_v, super_vertex);
        // MPI_Allreduce(MPI_IN_PLACE, super_vertex, vmin_edgeertices, MPI_TYP, MPI_SUM, MPI_COMM_WORLD); // could be better as less complex
        MPI_Reduce_scatter(MPI_IN_PLACE, super_vertex, recvCounts, MPI_TYP, MPI_SUM, MPI_COMM_WORLD);
        next_iter_v = -super_vertex[OLD_G->V - 1];                              // ensure correct size of next iteration
        mpi_exclusive_prefix_sum(start_v, stop_v, super_vertex, super_vertex);  // in place
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, super_vertex, recvCounts, displs, MPI_TYP, MPI_COMM_WORLD);

        FOR(i, start_v, stop_v) parent[i] = super_vertex[parent[i]];                                                        // improves locality, not sure if needed
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, parent, recvCounts, displs, MPI_TYP, MPI_COMM_WORLD, &req[0]);  // by inverting order we could make things better with igatherv

        /// NEW GRAPH
        // out edges
        next_iter_v += super_vertex[OLD_G->V - 1];
        if (next_iter_v <= 1) break;  // early exit here i suppose? -------------- not connected might pose a threat
        int *new_out_degree = calloc(next_iter_v, sizeof(int));
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);
        out_degree_init(OLD_G, parent, start_v, stop_v, &next_iter_e, new_out_degree);
        MPI_Iallreduce(MPI_IN_PLACE, &new_out_degree, next_iter_v, MPI_TYP, MPI_SUM, MPI_COMM_WORLD, &req[0]);
        MPI_Iallreduce(MPI_IN_PLACE, &next_iter_e, 1, MPI_TYP, MPI_SUM, MPI_COMM_WORLD, &req[1]);

        MPI_Waitall(2, req, MPI_STATUS_IGNORE);  // new_out_degree + next_iter_e
        // allocation
        NEW_G = allocate_new_graph(new_out_degree, next_iter_e, next_iter_v);
        compute_partitions(NEW_G, rank, commsz, &start_v, &stop_v, recvCounts, displs);  // new start + stop + recvCounts + displs

        // first edge
        mpi_exclusive_prefix_sum(start_v, stop_v, NEW_G->out_degree, NEW_G->first_edge);
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, NEW_G->first_edge, recvCounts, displs, MPI_TYP, MPI_COMM_WORLD, &req[0]);

        // edges
        reduce_edges(OLD_G, parent, super_vertex, edge_map, start_v, stop_v, NEW_G, edge_map_tmp);
        compute_start_stop_edges(NEW_G, start_v, stop_v, &start_e, &stop_e);
        compute_recvCounts_dipls(start_e, stop_e, rank, recvCounts1, displs1);
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, NEW_G->destination, recvCounts1, displs1, MPI_TYP, MPI_COMM_WORLD, &req[1]);
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, NEW_G->weight, recvCounts1, displs1, MPI_TYP, MPI_COMM_WORLD, &req[2]);
        MPI_Iallgatherv(edge_map_tmp, stop_e - start_e, MPI_INT, edge_map, recvCounts, displs, MPI_INT, MPI_COMM_WORLD, &req[3]);

        MPI_Waitall(3, req, MPI_STATUS_IGNORE);  // first_edge + destination + weight
        MPI_Wait(&req[3], MPI_STATUS_IGNORE);    // edge_map

        iteration++;
    }

    // MST gather
    compute_recvCounts_dipls(0, internal_index, rank, recvCounts, displs);
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

void compute_start_stop_vertices(const Graph_CSR *G, const int rank, const int commsz, TYP *start_v, TYP *stop_v) {
    int size = G->V;
    int partition = size / commsz;
    int missing = size % commsz;
    *start_v = rank * partition;
    *stop_v = *start_v + partition;
    if (rank < missing) {
        *start_v += partition;
    } else {
        *start_v += missing;
        *stop_v -= 1;
    }
}

void compute_start_stop_edges(const Graph_CSR *G, const TYP start_v, const TYP stop_v, TYP *start_e, TYP *stop_e) {
    *start_e = G->first_edge[start_v];
    *stop_e = G->first_edge[stop_v] + G->out_degree[stop_v];
}

void compute_recvCounts_dipls(const int start, const int stop, const int rank, int *recvCounts, int *displs) {
    recvCounts[rank] = stop - start;
    MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvCounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Exscan(recvCounts, displs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

void compute_partitions(const Graph_CSR *G, const int rank, const int commsz, TYP *start_v, TYP *stop_v, int *recvCounts, int *displs) {
    compute_start_stop_vertices(G, rank, commsz, start_v, stop_v);
    compute_recvCounts_dipls(*start_v, *stop_v, rank, recvCounts, displs);
}

void find_min_edge(const Graph_CSR *G, const int start_v, const int stop_v, TYP *min_edge) {
    int i;
    FOR(i, start_v, stop_v) {
        int start_e = G->first_edge[i];
        for (int j = start_e; j < start_e + G->out_degree[i]; ++j) {
            if (G->weight[j] < G->weight[min_edge[i]]) {
                min_edge[i] = j;
            }
        }
    }
}

void merge_edge(const Graph_CSR *G, const int *map_edge, const int start, const int stop, TYP *MST, TYP *min_edge, TYP *parent, int *internal_index) {
    int i;
    FOR(i, start, stop) {
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
}

void parent_compression(const int start, const int stop, TYP *parent) {
    int i, father, grand_father;
    FOR(i, start, stop) {
        do {
            parent[i] = parent[parent[i]];  // jump liv
            father = parent[i];
            grand_father = parent[father];
        } while (father != grand_father);
    }
}

void super_vertex_init(const TYP *parent, const TYP start, const TYP stop, TYP *super_vertex) {
    int i;
    FOR(i, start, stop)
    if (parent[i] != -1)  // if not connected
        super_vertex[parent[i]] = 1;
}

void mpi_exclusive_prefix_sum(const TYP start, const TYP stop, int *send, int *recv) {
    int i, local_sum = 0, offset = 0, tmp;

    FOR(i, start, stop) {  // Compute local prefix sum
        tmp = send[i];
        recv[i] = local_sum;
        local_sum += tmp;
    }

    MPI_Exscan(&local_sum, &offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);  // get offset

    FOR(i, start + 1, stop) recv[i] += offset;  // apply offset
}

void out_degree_init(const Graph_CSR *Gin, const TYP *parent, const TYP start, const TYP stop, int *next_iter_e, int *new_out_degree) {
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

Graph_CSR *allocate_new_graph(const int *out_degree, const int E, const int V) {
    Graph_CSR *G = base_graph(E, V);
    G->out_degree = out_degree;
    G->first_edge = malloc(V * sizeof(int));
    G->destination = malloc(E * sizeof(int));
    G->weight = malloc(E * sizeof(int));
    return G;
}

int last_binary_search(const TYP *super_vertex, const int size, const int lookfor) {
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

void reduce_edges(const Graph_CSR *Gin, const TYP *parent, const TYP *super_vertex, const int *edge_map, const TYP start, const TYP stop, Graph_CSR *Gout, int *edge_map_tmp) {
    int i, k, from, to, master_parent;
    int first_vertex_in = last_binary_search(super_vertex, Gin->V, start);  // cheap heuristic 1

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

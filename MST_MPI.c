#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "graph.h"

#define MAX_EDGES_FOR_VERTIX(vertix) ((vertix * (vertix - 1)) / 2)
#define FOR(i, start, stop) for (i = start; i <= stop; ++i)
#define OLD_G graph[iteration]
#define NEW_G graph[iteration + 1]

#define DEBUG 0

typedef int TYP;
#define MPI_TYP MPI_INT

void boruvka(const Graph_CSR *, const int, const int, TYP *, const MPI_Comm);

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, commsz;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsz);

    if (argc < 2) {
        if (!rank) perror("usage: mst_mpi <graph.txt>");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    //                   b  d  e  a  c  e  b  e  g  a  e  f  a  b  c  d  f  g  d  e  g  c  e  f
    // int destination[] = {1, 3, 4, 0, 2, 4, 1, 4, 6, 0, 4, 5, 0, 1, 2, 3, 5, 6, 3, 4, 6, 2, 4, 5};
    // thread      | 0               | 1                | 2                          | 3        |
    // vertex      | 0       | 1     | 2      | 3       | 4                | 5       | 6        |
    // chosen       0         3              8      10                16        19     21
    // int weight[] = {4, 12, 9, 4, 8, 7, 8, 5, 2, 12, 3, 6, 9, 7, 5, 3, 1, 13, 6, 1, 11, 2, 13, 11};
    // int first_edge[] = {0, 3, 6, 9, 12, 18, 21};
    // int out_degree[] = {3, 3, 3, 3, 6, 3, 3};
    // converter from other formats
    // Graph_CSR *G = default_graph(24, 7);
    // for (int i = 0; i < G->E; ++i) {
    //     G->destination[i] = destination[i];
    //     G->weight[i] = weight[i];
    // }
    // for (int i = 0; i < G->V; ++i) {
    //     G->first_edge[i] = first_edge[i];
    //     G->out_degree[i] = out_degree[i];
    // }

    Graph_CSR *G = NULL;
    TYP *MST = NULL;

    if (!rank) {
        G = init_graph_from_file(argv[1]);
        MST = malloc((G->V - 1) * sizeof(TYP));
        MPI_Bcast_GraphCSR(&G, MPI_COMM_WORLD);
    } else {
        MPI_Bcast_GraphCSR(&G, MPI_COMM_WORLD);
    }

    if (!rank && DEBUG) {
        int i;
        printf("Graph || E: %d | V: %d\n", G->E, G->V);
        printf("dest:   [ ");
        FOR(i, 0, G->E - 1) printf("%2d ", G->destination[i]);
        printf("]\nweight: [ ");
        FOR(i, 0, G->E - 1) printf("%2d ", G->weight[i]);
        printf("]\noutdeg: [ ");
        FOR(i, 0, G->V - 1) printf("%2d ", G->out_degree[i]);
        printf("]\nfiredg: [ ");
        FOR(i, 0, G->V - 1) printf("%2d ", G->first_edge[i]);
        printf("]\n");
    }

    boruvka(G, rank, commsz, MST, MPI_COMM_WORLD);

    if (!rank) {  // file? visual?
        int tot;
        printf("MST: [ ");
        for (int i = 0; i < G->V - 1; ++i) {
            tot += G->weight[MST[i]];
            printf("%d ", MST[i]);
        }
        printf("]\ntot= %d\n", tot);
    }

    if (!rank) free(MST);
    free_graph(G);
    MPI_Finalize();
    return 0;
}

void compute_start_stop(const int, const int, const int, TYP *, TYP *);
void compute_start_stop_edges(const Graph_CSR *, const TYP, const TYP, TYP *, TYP *);
void compute_recvCounts_dipls(const int, const int, const int, int *, int *, const MPI_Comm);
void compute_partitions(const Graph_CSR *, const int, const int, TYP *, TYP *, int *, int *, const MPI_Comm);
void find_min_edge(const Graph_CSR *, const int, const int, TYP *);
void merge_edge(const Graph_CSR *, const int *, const int, const int, TYP *, TYP *, TYP *, int *);
void parent_compression(const int, const int, TYP *);
void super_vertex_init(const TYP *, const TYP, const TYP, TYP *);
void mpi_exclusive_prefix_sum(const TYP, const TYP, const int, int *, int *, const MPI_Comm);
void out_degree_init(const Graph_CSR *, const TYP *, const TYP, const TYP, int *, int *);
Graph_CSR *allocate_new_graph(const int *, const int, const int);
void reduce_edges(const Graph_CSR *, const TYP *, const int *, const TYP, const TYP, const TYP, const TYP, Graph_CSR *, int *);

// LOGIC
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
void boruvka(const Graph_CSR *G, const int rank, const int commsz, TYP *MST, const MPI_Comm comm) {
    int iteration, next_iter_v, next_iter_e, internal_index = 0;
    TYP start_v, stop_v, start_e, stop_e, i;
    MPI_Request req[4];

    Graph_CSR **graph = (Graph_CSR **)malloc(((int)log2((float)G->V) + 1) * sizeof(Graph_CSR *));
    TYP *MST_tmp = malloc((G->V - 1) * sizeof(int));
    TYP *min_edge = malloc(G->V * sizeof(TYP));
    TYP *parent = malloc(G->V * sizeof(TYP));
    TYP *super_vertex = calloc(G->V, sizeof(TYP));
    int *recvCounts = calloc(commsz, sizeof(int));  // init to 0
    int *recvCounts1 = malloc(commsz * sizeof(int));
    int *displs = malloc(commsz * sizeof(int));
    int *displs1 = malloc(commsz * sizeof(int));
    int *edge_map = malloc(G->E * sizeof(int));
    int *edge_map_tmp = malloc(G->E * sizeof(int));

    // to implement if better
    // Graph_CSR **graph;
    // TYP *min_edge, *parent, *super_vertex;
    // int *recvCounts, *recvCounts1, *displs, *displs1;
    // int *edge_map, *edge_map_tmp;

    // MPI_Alloc_mem(((int)log2((float)G->V) + 1) * sizeof(Graph_CSR *), MPI_INFO_NULL, graph);
    // MPI_Alloc_mem(G->V * sizeofmin_edge(TYP), MPI_INFO_NULL, min_edge);
    // MPI_Alloc_mem(G->V * sizeof(TYP), MPI_INFO_NULL, parent);
    // MPI_Alloc_mem(G->V * sizeof(TYP), MPI_INFO_NULL, super_vertex);
    // MPI_Alloc_mem(commsz * sizeof(int), MPI_INFO_NULL, recvCounts);
    // MPI_Alloc_mem(commsz * sizeof(int), MPI_INFO_NULL, recvCounts1);
    // MPI_Alloc_mem(commsz * sizeof(int), MPI_INFO_NULL, displs);
    // MPI_Alloc_mem(commsz * sizeof(int), MPI_INFO_NULL, displs1);
    // MPI_Alloc_mem(G->E * sizeof(int), MPI_INFO_NULL, edge_map);
    // MPI_Alloc_mem(G->E * sizeof(int), MPI_INFO_NULL, edge_map_tmp);

    if (!(graph && min_edge && parent && super_vertex && recvCounts && recvCounts1 && displs && displs1 && edge_map && edge_map_tmp)) {
        perror("allocation error");
        MPI_Abort(comm, EXIT_FAILURE);
    }

    iteration = 0;
    OLD_G = G;

    compute_start_stop(OLD_G->V, rank, commsz, &start_v, &stop_v);
    if (DEBUG) printf("[%d] %d - %d\n", rank, start_v, stop_v);
    compute_start_stop(OLD_G->E, rank, commsz, &start_e, &stop_e);
    FOR(i, start_e, stop_e) edge_map[i] = i;
    compute_recvCounts_dipls(start_e, stop_e, rank, recvCounts, displs, comm);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, edge_map, recvCounts, displs, MPI_INT, comm);

    FOR(i, start_v, stop_v) min_edge[i] = -1;
    compute_recvCounts_dipls(start_v, stop_v, rank, recvCounts, displs, comm);

    while (OLD_G->V > 1) {
        // find min edgess for each v || uneven workload
        find_min_edge(OLD_G, start_v, stop_v, min_edge);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, min_edge, recvCounts, displs, MPI_TYP, comm);
        if (!rank && DEBUG) {
            printf("minedge: [ ");
            FOR(i, 0, OLD_G->V - 1) printf("%2d ", min_edge[i]);
            printf("]\n");
        }

        // remove mirrors + merge mfset
        merge_edge(OLD_G, edge_map, start_v, stop_v, MST_tmp, min_edge, parent, &internal_index);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, parent, recvCounts, displs, MPI_TYP, comm);
        if (!rank && DEBUG) {
            printf("parent:  [ ");
            FOR(i, 0, OLD_G->V - 1) printf("%2d ", parent[i]);
            printf("]\n");
        }

        // compression
        // the omp version should be better as there is the possibility o skipping more parents in the process as they are always updated
        // the mpi version has less possibility to skip the more the commsz increase but there is no overhead other that the all gatherv
        // hibrid seem difficult as there would be race conditions
        parent_compression(start_v, stop_v, parent);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, parent, recvCounts, displs, MPI_TYP, comm);
        if (!rank && DEBUG) {
            printf("compres: [ ");
            FOR(i, 0, OLD_G->V - 1) printf("%2d ", parent[i]);
            printf("]\n");
        }

        // supervertex init
        super_vertex_init(parent, start_v, stop_v, super_vertex);
        MPI_Allreduce(MPI_IN_PLACE, super_vertex, OLD_G->V, MPI_TYP, MPI_MAX, comm);  // could be better as less complex gather or init necessary
        // MPI_Reduce_scatter(MPI_IN_PLACE, super_vertex, recvCounts, MPI_TYP, MPI_SUM, MPI_COMM_WORLD);
        next_iter_v = -super_vertex[OLD_G->V - 1];  // ensure correct size of next iteration
        if (!rank && DEBUG) {
            printf("supvert: [ ");
            FOR(i, 0, OLD_G->V - 1) printf("%2d ", super_vertex[i]);
            printf("]\n");
        }

        mpi_exclusive_prefix_sum(start_v, stop_v, rank, super_vertex, super_vertex, comm);  // in place
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, super_vertex, recvCounts, displs, MPI_TYP, comm);
        if (!rank && DEBUG) {
            printf("exsupv:  [ ");
            FOR(i, 0, OLD_G->V - 1) printf("%2d ", super_vertex[i]);
            printf("]\n");
        }

        FOR(i, start_v, stop_v) parent[i] = super_vertex[parent[i]];                                              // improves locality, not sure if needed
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, parent, recvCounts, displs, MPI_TYP, comm, &req[0]);  // by inverting order we could make things better with igatherv

        // control
        next_iter_v += super_vertex[OLD_G->V - 1];

        /// NEW GRAPH
        // out edges
        int *new_out_degree = calloc(next_iter_v, sizeof(int));
        next_iter_e = 0;
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);  // parent
        if (!rank && DEBUG) {
            printf("suparnt: [ ");
            FOR(i, 0, OLD_G->V - 1) printf("%2d ", parent[i]);
            printf("]\nnext V:  [ %d ]\n", next_iter_v);
        }
        if (next_iter_v <= 1 || next_iter_v == OLD_G->V) break;  // early exit here i suppose? -------------- not connected might pose a threat (2 cond?)
        out_degree_init(OLD_G, parent, start_v, stop_v, &next_iter_e, new_out_degree);
        MPI_Iallreduce(MPI_IN_PLACE, new_out_degree, next_iter_v, MPI_TYP, MPI_SUM, comm, &req[0]);
        MPI_Iallreduce(MPI_IN_PLACE, &next_iter_e, 1, MPI_TYP, MPI_SUM, comm, &req[1]);

        MPI_Waitall(2, req, MPI_STATUS_IGNORE);  // new_out_degree + next_iter_e
        if (!rank && DEBUG) {
            printf("Next E:   [ %d ]\noutdeg:  [ ", next_iter_e);
            FOR(i, 0, next_iter_v - 1) printf("%2d ", new_out_degree[i]);
            printf("]\n");
        }
        // allocation
        NEW_G = allocate_new_graph(new_out_degree, next_iter_e, next_iter_v);
        compute_partitions(NEW_G, rank, commsz, &start_v, &stop_v, recvCounts, displs, comm);  // new start + stop + recvCounts + displs
        mpi_exclusive_prefix_sum(start_v, stop_v, rank, NEW_G->out_degree, NEW_G->first_edge, comm);
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, NEW_G->first_edge, recvCounts, displs, MPI_TYP, comm, &req[0]);
        // edges
        compute_start_stop_edges(NEW_G, start_v, stop_v, &start_e, &stop_e);
        reduce_edges(OLD_G, parent, edge_map, start_v, stop_v, start_e, stop_e, NEW_G, edge_map_tmp);
        compute_recvCounts_dipls(start_e, stop_e, rank, recvCounts1, displs1, comm);
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, NEW_G->destination, recvCounts1, displs1, MPI_TYP, comm, &req[1]);
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, NEW_G->weight, recvCounts1, displs1, MPI_TYP, comm, &req[2]);
        if (DEBUG) {
            printf("[ %d | %d %d | %2d %2d | %2d ] rcv: [ ", rank, start_v, stop_v, start_e, stop_e, stop_e - start_e + 1);
            FOR(i, 0, commsz - 1) printf("%d ", recvCounts1[i]);
            printf("] | dsps: [ ");
            FOR(i, 0, commsz - 1) printf("%d ", displs1[i]);
            printf("] | tmp: [ ");
            FOR(i, 1, start_e) printf("  __ ");
            FOR(i, start_e, stop_e) printf("%4d ", edge_map_tmp[i]);
            FOR(i, stop_e + 1, NEW_G->E - 1) printf(" ___ ");
            printf("]\n");
        }
        MPI_Iallgatherv(&edge_map_tmp[start_e], stop_e - start_e + 1, MPI_INT, edge_map, recvCounts1, displs1, MPI_INT, comm, &req[3]);
        FOR(i, 0, NEW_G->V - 1) super_vertex[i] = 0;  // reset
        MPI_Waitall(4, req, MPI_STATUS_IGNORE);       // first_edge + destination + weight + edge_map
        if (!rank && DEBUG) {
            printf("edge_map: [ ");
            FOR(i, 0, NEW_G->E - 1) printf("%2d ", edge_map[i]);
            printf("]\nNEW_G || E: %d | V: %d\n", NEW_G->E, NEW_G->V);
            printf("dest:   [ ");
            FOR(i, 0, NEW_G->E - 1) printf("%2d ", NEW_G->destination[i]);
            printf("]\nweight: [ ");
            FOR(i, 0, NEW_G->E - 1) printf("%2d ", NEW_G->weight[i]);
            printf("]\noutdeg: [ ");
            FOR(i, 0, NEW_G->V - 1) printf("%2d ", NEW_G->out_degree[i]);
            printf("]\nfiredg: [ ");
            FOR(i, 0, NEW_G->V - 1) printf("%2d ", NEW_G->first_edge[i]);
            printf("]\n");
        }

        iteration++;
    }

    if (DEBUG) {
        printf("[%d] intind: %d [ ", rank, internal_index);
        FOR(i, 0, internal_index - 1) printf("%d ", MST_tmp[i]);
        printf("]\n");
    }
    // MST gather
    compute_recvCounts_dipls(0, internal_index - 1, rank, recvCounts, displs, comm);
    MPI_Igatherv(MST_tmp, internal_index, MPI_TYP, MST, recvCounts, displs, MPI_TYP, 0, comm, &req[0]);

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
    free(MST_tmp);

    MPI_Wait(&req[0], MPI_STATUS_IGNORE);
}

void compute_start_stop(const int size, const int rank, const int commsz, TYP *start_v, TYP *stop_v) {
    int partition = size / commsz;
    int missing = size % commsz;

    if (rank < missing) {
        *start_v = rank * (partition + 1);
        *stop_v = *start_v + partition;
    } else if (partition) {
        *start_v = rank * partition + missing;
        *stop_v = *start_v + partition - 1;
    } else {  // useless process
        *start_v = 1;
        *stop_v = 0;
    }
}

void compute_start_stop_edges(const Graph_CSR *G, const TYP start_v, const TYP stop_v, TYP *start_e, TYP *stop_e) {
    if (start_v <= stop_v) {
        *start_e = G->first_edge[start_v];
        *stop_e = G->first_edge[stop_v] + G->out_degree[stop_v] - 1;
    } else {
        *start_e = 1;
        *stop_e = 0;
    }
}

void compute_recvCounts_dipls(const int start, const int stop, const int rank, int *recvCounts, int *displs, const MPI_Comm comm) {
    recvCounts[rank] = stop - start + 1;
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvCounts, 1, MPI_INT, comm);
    MPI_Exscan(&recvCounts[rank], &displs[rank], 1, MPI_INT, MPI_SUM, comm);
    displs[0] = 0;
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, displs, 1, MPI_INT, comm);
}

void compute_partitions(const Graph_CSR *G, const int rank, const int commsz, TYP *start_v, TYP *stop_v, int *recvCounts, int *displs, const MPI_Comm comm) {
    compute_start_stop(G->V, rank, commsz, start_v, stop_v);
    compute_recvCounts_dipls(*start_v, *stop_v, rank, recvCounts, displs, comm);
}

void find_min_edge(const Graph_CSR *G, const int start_v, const int stop_v, TYP *min_edge) {
    int i, j, start_e, stop_e;
    FOR(i, start_v, stop_v) {
        compute_start_stop_edges(G, i, i, &start_e, &stop_e);
        min_edge[i] = start_e;
        FOR(j, start_e + 1, stop_e) {
            if (G->weight[j] < G->weight[min_edge[i]] || (G->weight[j] == G->weight[min_edge[i]] && G->destination[j] < G->destination[min_edge[i]])) {
                min_edge[i] = j;
            }
        }
    }
}

void merge_edge(const Graph_CSR *G, const int *map_edge, const int start_v, const int stop_v, TYP *MST_tmp, TYP *min_edge, TYP *parent, int *internal_index) {
    int i, dest, dest_dest;
    FOR(i, start_v, stop_v) {
        dest = G->destination[min_edge[i]];                        // i -> destination
        dest_dest = G->destination[min_edge[dest]];                // destination -> destination
        if (dest >= 0 && (dest_dest != i || dest < i)) {           // if init + if mirror only save the one pointing to the smalles one
            parent[i] = dest;                                      // mfset init
            MST_tmp[(*internal_index)++] = map_edge[min_edge[i]];  // MST store using edge_map ---------------doubling edges?
        } else {                                                   //
            parent[i] = i;                                         // mfset init
        }
        min_edge[i] = -1;  // min_edge reset for next iter
    }
}

void parent_compression(const int start_v, const int stop_v,
                        TYP *parent) {  // spinlock if all edges arent mirrored and if there is no condition on equal weights (done condition in min_edge, still missing verification on double edges)
    int i, father, grand_father;
    FOR(i, start_v, stop_v) {
        do {
            parent[i] = parent[parent[i]];  // jump liv
            father = parent[i];
            grand_father = parent[father];
        } while (father != grand_father);
    }
}

void super_vertex_init(const TYP *parent, const TYP start_v, const TYP stop_v, TYP *super_vertex) {  // replace in main?
    int i;
    FOR(i, start_v, stop_v) super_vertex[parent[i]] = (parent[i] != -1);
}

void mpi_exclusive_prefix_sum(const TYP start, const TYP stop, const int rank, int *send, int *recv, const MPI_Comm comm) {
    int i, local_sum = 0, offset = 0, tmp;

    FOR(i, start, stop) {  // Compute local prefix sum
        tmp = send[i];
        recv[i] = local_sum;
        local_sum += tmp;
    }

    MPI_Exscan(&local_sum, &offset, 1, MPI_INT, MPI_SUM, comm);
    if (rank > 0) FOR(i, start, stop) recv[i] += offset;
}

void out_degree_init(const Graph_CSR *Gin, const TYP *parent, const TYP start_v, const TYP stop_v, int *next_iter_e, int *new_out_degree) {  // can be restructured to be more efficient?
    int i, ii, start_e, stop_e;
    FOR(i, start_v, stop_v) {
        compute_start_stop_edges(Gin, i, i, &start_e, &stop_e);
        FOR(ii, start_e, stop_e) {
            if (parent[Gin->destination[ii]] != parent[i]) {
                new_out_degree[parent[i]] += 1;
                *next_iter_e += 1;
            }
        }
    }
}

Graph_CSR *allocate_new_graph(const int *out_degree, const int E, const int V) {  // in graph.c?
    Graph_CSR *G = base_graph(E, V);
    if (!G) printf("allocation err\n");
    G->out_degree = out_degree;
    G->first_edge = malloc(V * sizeof(int));
    G->destination = malloc(E * sizeof(int));
    G->weight = malloc(E * sizeof(int));
    return G;
}

// int last_binary_search(const TYP *super_vertex, const int size, const int lookfor) {
//     // if (DEBUG) printf("looking for:%d\n", lookfor);
//     int left = 0, right = size - 1, mid;
//     while (left < right) {
//         mid = (left + right) / 2;
//         if (super_vertex[mid] < lookfor)
//             left = mid + 1;
//         else
//             right = mid;
//     }
//     // if (DEBUG) printf("looking for:%d found:%d\n", lookfor, left);
//     return right;
// }

void reduce_edges(const Graph_CSR *Gin, const TYP *parent, const int *edge_map, const TYP start_v, const TYP stop_v, const TYP start_e, const TYP stop_e, Graph_CSR *Gout,
                  int *edge_map_tmp) {  // verify if the heuristic can be triggered
    int i, k, edge_out = start_e, from, to, master_parent;
    FOR(i, 0, Gin->V - 1) {  // for each possible vertex in Gin
        master_parent = parent[i];
        if (master_parent >= start_v && master_parent <= stop_v) {  // if vertex parent is within range
            compute_start_stop_edges(Gin, i, i, &from, &to);
            FOR(k, from, to) {
                if (parent[Gin->destination[k]] != master_parent) {             // if edge cross parent
                    Gout->destination[edge_out] = parent[Gin->destination[k]];  // add edge
                    Gout->weight[edge_out] = Gin->weight[k];                    //
                    edge_map_tmp[edge_out] = edge_map[k];                       // sync mapping
                    edge_out++;                                                 // increment counter
                }
            }
            if (edge_out > stop_e) break;  // cheap heuristic
        }
    }
}

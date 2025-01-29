#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "graph.h"

#define FOR(i, start, stop) for (i = start; i <= stop; ++i)
#define OLD_G graph[iteration]
#define NEW_G graph[iteration + 1]

#define DEBUG 1

typedef int TYP;
#define MPI_TYP MPI_INT

void boruvka(Graph_CSR *, const int, const int, TYP *, const MPI_Comm);

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, commsz;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsz);

    if (argc < 2) {
        if (!rank) perror("usage: mst_mpi <graph.txt>");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    Graph_CSR *G = NULL;
    TYP *MST = NULL;
    int MST_size;

    if (!rank) {
        G = MPI_init_graph_from_file(argv[1]);
        MST_size = G->V - 1;
        MPI_Alloc_mem(MST_size * sizeof(TYP), MPI_INFO_NULL, &MST);
    }

    double t = MPI_Wtime();
    boruvka(G, rank, commsz, MST, MPI_COMM_WORLD);
    t = MPI_Wtime() - t;

    if (!rank) {  // file? visual?
        long int tot = 0;
        printf("\rMST: [ ");
        for (int i = 0; i < MST_size; ++i) {
            tot += G->weight[MST[i]];
            // printf("%d ", MST[i]);
        }
        printf("]\ntot= %ld\n", tot);
        printf("time: %f\n", t);
        MPI_Free_mem(MST);
        MPI_Free_mem(G);
    }

    MPI_Finalize();
    return 0;
}

void verify_mirror(const Graph_CSR *);
void compute_start_stop(const int, const int, const int, TYP *, TYP *);
void compute_start_stop_edges(const Graph_CSR *, const TYP, const TYP, TYP *, TYP *);
void compute_recvCounts_dipls(const int, const int, const int, int *, int *, const MPI_Comm);
void compute_partitions(const Graph_CSR *, const int, const int, TYP *, TYP *, int *, int *, const MPI_Comm);
void find_min_edge(const Graph_CSR *, const int, const int, TYP *);
void merge_edge(const Graph_CSR *, const int *, const int, const int, TYP *, TYP *, TYP *, int *);
void parent_compression(const int, const int, TYP *);
void super_vertex_init(const TYP *, const int, const TYP, const TYP, TYP *);
void mpi_exclusive_prefix_sum(const TYP, const TYP, const int, int *, int *, const MPI_Comm);
void out_degree_init(const Graph_CSR *, const TYP *, const int, const TYP, const TYP, int *, int *);
void reduce_edges(const Graph_CSR *, const TYP *, const int *, const TYP, const TYP, Graph_CSR *, int *);

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
void boruvka(Graph_CSR *G, const int rank, const int commsz, TYP *MST, const MPI_Comm comm) {
    MPI_Bcast_GraphCSR(&G, MPI_COMM_WORLD);

    int iteration, next_iter_v, next_iter_e, internal_index = 0;
    TYP start_v, stop_v, start_e, stop_e, i;
    MPI_Request req[4];

    Graph_CSR **graph;
    TYP *min_edge, *parent, *super_vertex, *MST_tmp;
    int *recvCounts, *recvCounts1, *displs, *displs1;
    int *edge_map, *edge_map_tmp;
    int *new_out_degree;

    MPI_Alloc_mem(((int)log2((float)G->V) + 1) * sizeof(Graph_CSR *), MPI_INFO_NULL, &graph);
    MPI_Alloc_mem(G->V * sizeof(TYP), MPI_INFO_NULL, &min_edge);
    MPI_Alloc_mem(G->V * sizeof(TYP), MPI_INFO_NULL, &parent);
    MPI_Alloc_mem(G->V * sizeof(TYP), MPI_INFO_NULL, &super_vertex);
    MPI_Alloc_mem(G->V * sizeof(TYP), MPI_INFO_NULL, &MST_tmp);
    MPI_Alloc_mem(commsz * sizeof(int), MPI_INFO_NULL, &recvCounts);
    MPI_Alloc_mem(commsz * sizeof(int), MPI_INFO_NULL, &recvCounts1);
    MPI_Alloc_mem(commsz * sizeof(int), MPI_INFO_NULL, &displs);
    MPI_Alloc_mem(commsz * sizeof(int), MPI_INFO_NULL, &displs1);
    MPI_Alloc_mem(G->E * sizeof(int), MPI_INFO_NULL, &edge_map);
    MPI_Alloc_mem(G->E * sizeof(int), MPI_INFO_NULL, &edge_map_tmp);

    if (!(graph && min_edge && parent && super_vertex && recvCounts && recvCounts1 && displs && displs1 && edge_map && edge_map_tmp)) {
        perror("allocation error");
        MPI_Abort(comm, EXIT_FAILURE);
    }

    iteration = 0;
    OLD_G = G;

    compute_start_stop(OLD_G->V, rank, commsz, &start_v, &stop_v);
#if DEBUG
    printf("[%d] %d - %d\n", rank, start_v, stop_v);
#endif

    compute_start_stop(OLD_G->E, rank, commsz, &start_e, &stop_e);
    FOR(i, start_e, stop_e) edge_map[i] = i;
    compute_recvCounts_dipls(start_e, stop_e, rank, recvCounts, displs, comm);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, edge_map, recvCounts, displs, MPI_INT, comm);

    FOR(i, start_v, stop_v) min_edge[i] = -1;
    compute_recvCounts_dipls(start_v, stop_v, rank, recvCounts, displs, comm);

    while (OLD_G->V > 1) {
#if DEBUG
        if (!rank) {
            verify_mirror(OLD_G);

            printf("Graph || E: %d | V: %d\n", OLD_G->E, OLD_G->V);
            printf("dest:   [ ");
            FOR(i, 0, OLD_G->E - 1) printf("%2d ", OLD_G->destination[i]);
            printf("]\nweight: [ ");
            FOR(i, 0, OLD_G->E - 1) printf("%2d ", OLD_G->weight[i]);
            printf("]\noutdeg: [ ");
            FOR(i, 0, OLD_G->V - 1) printf("%2d ", OLD_G->out_degree[i]);
            printf("]\nfiredg: [ ");
            FOR(i, 0, OLD_G->V - 1) printf("%2d ", OLD_G->first_edge[i]);
            printf("]\n");
        }
#else
        if (!rank) {
            printf("\r%2d %6d", iteration, OLD_G->E);
            fflush(stdout);
        }
#endif

        // find min edgess for each v || uneven workload
        find_min_edge(OLD_G, start_v, stop_v, min_edge);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, min_edge, recvCounts, displs, MPI_TYP, comm);
#if DEBUG
        if (!rank) {
            // int a = 0, b = 10;
            // printf("edges [%d]\n", a);
            // FOR(i, OLD_G->first_edge[a], OLD_G->first_edge[a + 1] - 1) if (OLD_G->destination[i] == b) printf("[%d - %d]\n", OLD_G->destination[i], OLD_G->weight[i]);
            // printf("edges [%d]\n", b);
            // FOR(i, OLD_G->first_edge[b], OLD_G->first_edge[b + 1] - 1) if (OLD_G->destination[i] == a) printf("[%d - %d]\n", OLD_G->destination[i], OLD_G->weight[i]);
            printf("minedge: [ ");
            FOR(i, 0, OLD_G->V - 1) printf("%4d ", min_edge[i]);
            printf("]\n");
            // printf("minedge ed_w: [ ");
            // FOR(i, 0, OLD_G->V - 1) printf("%4d ", OLD_G->weight[min_edge[i]]);
            // printf("]\n");
            // printf("minedge v_in: [ ");
            // FOR(i, 0, OLD_G->V - 1) printf("%4d ", i);
            // printf("]\n");
            // printf("minedge v_ou: [ ");
            // FOR(i, 0, OLD_G->V - 1) printf("%4d ", OLD_G->destination[min_edge[i]]);
            // printf("]\n");
        }
#endif

        // remove mirrors + merge mfset
        merge_edge(OLD_G, edge_map, start_v, stop_v, MST_tmp, min_edge, parent, &internal_index);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, parent, recvCounts, displs, MPI_TYP, comm);
#if DEBUG
        if (!rank) {
            printf("parent:  [ ");
            FOR(i, 0, OLD_G->V - 1) printf("%2d ", parent[i]);
            printf("]\n");
        }
#endif

        // compression
        // the omp version should be better as there is the possibility o skipping more parents in the process as they are always updated
        // the mpi version has less possibility to skip the more the commsz increase but there is no overhead other that the all gatherv
        // hibrid seem difficult as there would be race conditions
        parent_compression(start_v, stop_v, parent);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, parent, recvCounts, displs, MPI_TYP, comm);
#if DEBUG
        if (!rank) {
            printf("compres: [ ");
            FOR(i, 0, OLD_G->V - 1) printf("%2d ", parent[i]);
            printf("]\n");
        }
#endif

        // supervertex init
        super_vertex_init(parent, OLD_G->V, start_v, stop_v, super_vertex);
        MPI_Allreduce(MPI_IN_PLACE, super_vertex, OLD_G->V, MPI_TYP, MPI_MAX, comm);  // could be better as less complex gather or init necessary
        // MPI_Reduce_scatter(MPI_IN_PLACE, super_vertex, recvCounts, MPI_TYP, MPI_SUM, MPI_COMM_WORLD);
        next_iter_v = -super_vertex[OLD_G->V - 1];  // ensure correct size of next iteration
#if DEBUG
        if (!rank) {
            printf("supvert: [ ");
            FOR(i, 0, OLD_G->V - 1) printf("%2d ", super_vertex[i]);
            printf("]\n");
        }
#endif

        mpi_exclusive_prefix_sum(start_v, stop_v, rank, super_vertex, super_vertex, comm);  // in place
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, super_vertex, recvCounts, displs, MPI_TYP, comm);
#if DEBUG
        if (!rank) {
            printf("exsupv:  [ ");
            FOR(i, 0, OLD_G->V - 1) printf("%2d ", super_vertex[i]);
            printf("]\n");
        }
#endif

        // parent normalization
        FOR(i, start_v, stop_v) parent[i] = super_vertex[parent[i]];                                              // improves locality, not sure if needed
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, parent, recvCounts, displs, MPI_TYP, comm, &req[0]);  // by inverting order we could make things better with igatherv
        next_iter_v += super_vertex[OLD_G->V - 1];

        /// NEW GRAPH
        // out edges
        MPI_Alloc_mem(next_iter_v * sizeof(int), MPI_INFO_NULL, &new_out_degree);
        next_iter_e = 0;
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);  // parent
#if DEBUG
        if (!rank) {
            printf("suparnt: [ ");
            FOR(i, 0, OLD_G->V - 1) printf("%2d ", parent[i]);
            printf("]\nnext V:  [ %d ]\n", next_iter_v);
        }
#endif

        // control
        if (next_iter_v <= 1 || next_iter_v == OLD_G->V) break;  // early exit here i suppose? -------------- not connected might pose a threat (2 cond?)
        out_degree_init(OLD_G, parent, next_iter_v, start_v, stop_v, &next_iter_e, new_out_degree);
        MPI_Iallreduce(MPI_IN_PLACE, new_out_degree, next_iter_v, MPI_TYP, MPI_SUM, comm, &req[0]);
        MPI_Iallreduce(MPI_IN_PLACE, &next_iter_e, 1, MPI_TYP, MPI_SUM, comm, &req[1]);

        MPI_Waitall(2, req, MPI_STATUS_IGNORE);  // new_out_degree + next_iter_e
#if DEBUG
        if (!rank) {
            printf("Next E:   [ %d ]\noutdeg:  [ ", next_iter_e);
            FOR(i, 0, next_iter_v - 1) printf("%2d ", new_out_degree[i]);
            printf("]\n");
        }
#endif

        // allocation
        NEW_G = MPI_allocate_new_graph(new_out_degree, next_iter_e, next_iter_v);
        compute_partitions(NEW_G, rank, commsz, &start_v, &stop_v, recvCounts, displs, comm);  // new start + stop + recvCounts + displs
        mpi_exclusive_prefix_sum(start_v, stop_v, rank, NEW_G->out_degree, NEW_G->first_edge, comm);
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, NEW_G->first_edge, recvCounts, displs, MPI_TYP, comm, &req[0]);
#if DEBUG
        if (!rank) printf("allocated new graph\n");
#endif
        // edges
        compute_start_stop_edges(NEW_G, start_v, stop_v, &start_e, &stop_e);
        MPI_Wait(&req[0], MPI_STATUS_IGNORE);  // first_edge
        reduce_edges(OLD_G, parent, edge_map, start_v, stop_v, NEW_G, edge_map_tmp);
        if (!rank) printf("reduce edges");
        if (!(!rank && iteration == 0)) MPI_free_graph(OLD_G);
        compute_recvCounts_dipls(start_e, stop_e, rank, recvCounts1, displs1, comm);
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, NEW_G->destination, recvCounts1, displs1, MPI_TYP, comm, &req[0]);
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, NEW_G->weight, recvCounts1, displs1, MPI_TYP, comm, &req[1]);
#if DEBUG
        printf("[ %d | %d %d | %2d %2d | %2d ] rcv: [ ", rank, start_v, stop_v, start_e, stop_e, stop_e - start_e + 1);
        FOR(i, 0, commsz - 1) printf("%d ", recvCounts1[i]);
        printf("] | dsps: [ ");
        FOR(i, 0, commsz - 1) printf("%d ", displs1[i]);
        printf("] | tmp: [ ");
        FOR(i, 1, start_e) printf("  __ ");
        FOR(i, start_e, stop_e) printf("%4d ", edge_map_tmp[i]);
        FOR(i, stop_e + 1, NEW_G->E - 1) printf(" ___ ");
        printf("]\n");
#endif

        MPI_Iallgatherv(&edge_map_tmp[start_e], stop_e - start_e + 1, MPI_INT, edge_map, recvCounts1, displs1, MPI_INT, comm, &req[2]);
        MPI_Waitall(3, req, MPI_STATUS_IGNORE);  // destination + weight + edge_map
#if DEBUG
        if (!rank) {
            printf("edge_map: [ ");
            FOR(i, 0, NEW_G->E - 1) printf("%2d ", edge_map[i]);
            printf("]\n");
        }
#endif

        iteration++;
    }

#if DEBUG
    printf("[%d] intind: %d [ ", rank, internal_index);
    FOR(i, 0, internal_index - 1) printf("%d ", MST_tmp[i]);
    printf("]\n");
#endif

    // MST gather
    compute_recvCounts_dipls(0, internal_index - 1, rank, recvCounts, displs, comm);
    MPI_Igatherv(MST_tmp, internal_index, MPI_TYP, MST, recvCounts, displs, MPI_TYP, 0, comm, &req[0]);

    MPI_Free_mem(OLD_G);
    MPI_Free_mem(edge_map_tmp);
    MPI_Free_mem(edge_map);
    MPI_Free_mem(displs1);
    MPI_Free_mem(displs);
    MPI_Free_mem(recvCounts1);
    MPI_Free_mem(recvCounts);
    MPI_Free_mem(MST_tmp);
    MPI_Free_mem(super_vertex);
    MPI_Free_mem(parent);
    MPI_Free_mem(min_edge);
    MPI_Free_mem(graph);

    MPI_Wait(&req[0], MPI_STATUS_IGNORE);
}

void verify_mirror(const Graph_CSR *G) {
    int *visited;
    MPI_Alloc_mem(G->E * sizeof(int), MPI_INFO_NULL, &visited);
    for (int i = 0; i < G->E; ++i) visited[i] = 0;

    int w, found;
    for (int i = 0; i < G->V; ++i) {
        for (int j = G->first_edge[i]; j < G->first_edge[i + 1]; ++i) {
            if (!visited[j]) {
                visited[j] = 1;
                w = G->weight[i];
                found = 0;
                for (int k = G->first_edge[G->destination[j]]; k < G->first_edge[G->destination[j] + 1]; ++k) {
                    if (G->destination[k] == i) {
                        visited[k] = 1;
                        found = 1;
                        break;
                    }
                }
                if (!found) break;
            }
        }
        if (!found) break;
    }

    printf("mirror c: %d ", found);
    for (int i = 0; i < G->E; ++i) visited[i] = 0;

    for (int i = 0; i < G->V; ++i) {
        for (int j = G->first_edge[i]; j < G->first_edge[i + 1]; ++i) {
            if (!visited[j]) {
                visited[j] = 1;
                w = G->weight[i];
                found = 0;
                for (int k = G->first_edge[G->destination[j]]; k < G->first_edge[G->destination[j] + 1]; ++k) {
                    if (G->destination[k] == i && G->weight[k] == w) {
                        visited[k] = 1;
                        found = 1;
                        break;
                    }
                }
                if (!found) break;
            }
        }
        if (!found) break;
    }

    printf("w: %d ", found);
    for (int i = 0; i < G->E; ++i) visited[i] = 0;

    for (int i = 0; i < G->V; ++i) {
        for (int j = G->first_edge[i]; j < G->first_edge[i + 1]; ++i) {
            if (!visited[j]) {
                visited[j] = 1;
                w = G->weight[i];
                found = 0;
                for (int k = G->first_edge[G->destination[j]]; k < G->first_edge[G->destination[j] + 1]; ++k) {
                    if (!visited[k] && G->destination[k] == i) {
                        visited[k] = 1;
                        found = 1;
                        break;
                    }
                }
                if (!found) break;
            }
        }
        if (!found) break;
    }

    printf("n: %d\n", found);
    free(visited);
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
        parent[i] = i;                                                 // mfset init
        if (min_edge[i] >= 0) {                                        //
            dest = G->destination[min_edge[i]];                        // i -> destination
            dest_dest = G->destination[min_edge[dest]];                // destination -> destination
            if (dest_dest != i || dest < i) {                          // if mirror only save the one pointing to the smalles one
                parent[i] = dest;                                      // mfset init
                MST_tmp[(*internal_index)++] = map_edge[min_edge[i]];  // MST store using edge_map ---------------doubling edges?
            }
            min_edge[i] = -1;  // min_edge reset for next iter verify if real
        }
    }
}

void parent_compression(const int start_v, const int stop_v, TYP *parent) {  // handled spinlocks?
    int i, father, grand_father;
    FOR(i, start_v, stop_v) {
        do {
            parent[i] = parent[parent[i]];  // jump liv
            father = parent[i];
            grand_father = parent[father];
        } while (father != grand_father);
    }
}

void super_vertex_init(const TYP *parent, const int V, const TYP start_v, const TYP stop_v, TYP *super_vertex) {
    int i;
    FOR(i, 0, V - 1) super_vertex[i] = 0;                                 // reset
    FOR(i, start_v, stop_v) super_vertex[parent[i]] = (parent[i] != -1);  // partial init
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

void out_degree_init(const Graph_CSR *Gin, const TYP *parent, const int next_iter_v, const TYP start_v, const TYP stop_v, int *next_iter_e, int *new_out_degree) {  // can be restructured?
    int i, ii, start_e, stop_e, master_parent;

    FOR(i, 0, next_iter_v - 1) new_out_degree[i] = 0;  // reset

    FOR(i, start_v, stop_v) {
        master_parent = parent[i];
        compute_start_stop_edges(Gin, i, i, &start_e, &stop_e);
        FOR(ii, start_e, stop_e) {
            if (parent[Gin->destination[ii]] != master_parent) {
                new_out_degree[master_parent] += 1;
                *next_iter_e += 1;
            }
        }
    }
}

void reduce_edges(const Graph_CSR *Gin, const TYP *parent, const int *edge_map, const TYP start_v, const TYP stop_v, Graph_CSR *Gout, int *edge_map_tmp) {
    int super_parent, i, j, from, to, *index;

    MPI_Alloc_mem((stop_v - start_v + 1) * sizeof(int), MPI_INFO_NULL, &index);  // alloc
    int *offset_index = index - start_v;                                         // offset optimization
    FOR(i, start_v, stop_v) offset_index[i] = Gout->first_edge[i];               // init

    FOR(i, 0, Gin->V - 1) {                                                                       //              all vertices
        super_parent = parent[i];                                                                 //
        if (super_parent >= start_v && super_parent <= stop_v) {                                  //              all vertices assigned to this process
            compute_start_stop_edges(Gin, i, i, &from, &to);                                      //
            FOR(j, from, to) {                                                                    // all edges of all vertices assigned to this process
                if (parent[Gin->destination[j]] != super_parent) {                                // all edges of all vertices assigned to this process that connect two parents
                    Gout->destination[offset_index[super_parent]] = parent[Gin->destination[j]];  // add edge to gout
                    Gout->weight[offset_index[super_parent]] = Gin->weight[j];                    //
                    edge_map_tmp[offset_index[super_parent]] = edge_map[j];                       // sync edge_map
                    offset_index[super_parent]++;                                                 // increment index
                }
            }
        }
    }
    MPI_Free_mem(index);
}
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "graph.h"

#define FOR(start, stop) for (i = start; i < stop; ++i)

#define INPUT "graph.txt"
#define OUTPUT "mst.txt"

typedef int TYP;
#define MPI_TYP MPI_INT

#define MAX_W 1000
#define MAX_EDGES_FOR_VERTIX(vertix) ((vertix * (vertix - 1)) / 2)

void boruvka(const Graph_CSR *, TYP *, int, int);

int main(int argc, char **argv) {
    MPI_Init(argc, argv);

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
void boruvka(const Graph_CSR *G, TYP *MST, int rank, int commsz) {
    int iteration = 0, removed;
    int next_iter_v, next_iter_e;
    Graph_CSR **pastG = (Graph_CSR **)malloc(((int)log2f((float)G->V) + 1) * sizeof(Graph_CSR *));
    pastG[iteration] = G;
    TYP start, stop, partition, missing, i, count;
    // sizes never full used coul be / commsz but then allgatherv wouldnt work, i leave it there as of now
    TYP *min_edge = malloc(G->V * sizeof(TYP));  // if multiple of rank -> less overhead
    TYP *parent = malloc(G->V * sizeof(TYP));
    TYP *super_vertex = calloc(G->V, sizeof(TYP));
    int *recvCounts = malloc(commsz * sizeof(int));
    int *displs = malloc(commsz * sizeof(int));

    TYP vertices;
    while (vertices > 0) {
        removed = 0;
        vertices = pastG[iteration]->V;
        // partition define as well as buffers for gatherv
        partition = vertices / commsz;
        missing = vertices % commsz;
        start = rank * partition;
        stop = start + partition;
        if (rank < missing) {
            start += partition;
        } else {
            start += missing;
            stop -= 1;
        }
        count = partition + 1;

        for (i = 0; i < missing; ++i) {
            recvCounts[i] = count;
            displs[i] = count * i;
        }
        for (int j = 0; i < commsz; ++i, ++j) {
            recvCounts[i] = partition;
            displs[i] = count * missing + partition * j;
        }
        // ^^^^^^^^^^^^^^^^^^^^ POSSIBLE to parallelize this between processes, but it would be such a bother that the overhead would make it usless

        // init parent
        FOR(start, stop) parent[i] = -1;

        // find min edgess for each v || uneven workload
        FOR(start, stop) find_min_edge(pastG[iteration], min_edge, i);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, min_edge, recvCounts, displs, MPI_TYP, MPI_COMM_WORLD);

        // remove mirrors
        // merge mfset
        FOR(start, stop) merge_edges(pastG[iteration], MST, min_edge, parent, i, removed);

        // here i should save the edges i want to put in mst
        // for now we save partial in mst and final gather will do the trick

        MPI_Request req[4];

        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, parent, recvCounts, displs, MPI_TYP, MPI_COMM_WORLD, &req[0]);
        MPI_Iallreduce(MPI_IN_PLACE, &removed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &req[1]);

        MPI_Iallreduce(MPI_IN_PLACE, MST, G->V, MPI_TYP, MPI_SUM, MPI_COMM_WORLD, &req[2]);                                   // mst must be init 0
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, min_edge, recvCounts, displs, MPI_TYP, MPI_COMM_WORLD, &req[3]);  // could be usless

        MPI_Wait(&req[0], MPI_STATUS_IGNORE);  // parent

        // compression
        // the omp version should be better as there is the possibility o skipping more parents in the process as they are always updated
        // the mpi version has less possibility to skip the more the commsz increasebut there is no overhead other that the all gatherv
        // hibrid seem difficult as there would be race conditions
        FOR(start, stop) parent_compression(parent, i);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, parent, recvCounts, displs, MPI_TYP, MPI_COMM_WORLD);

        // vertices and edges

        MPI_Wait(&req[1], MPI_STATUS_IGNORE);  // removed
        // new graph
        next_iter_e = pastG[iteration]->E - removed * 2;
        next_iter_v = pastG[iteration]->V - removed;
        pastG[iteration + 1] = default_graph(next_iter_e, next_iter_v);

        // exclusive prefixsum
        super_vertex_init(super_vertex, parent, start, stop);

        // MPI_Allreduce(MPI_IN_PLACE, super_vertex, vertices, MPI_TYP, MPI_SUM, MPI_COMM_WORLD); // could be better as less complex
        MPI_Reduce_scatter(MPI_IN_PLACE, super_vertex, recvCounts, MPI_TYP, MPI_SUM, MPI_COMM_WORLD);

        mpi_exclusive_prefix_sum(super_vertex, super_vertex, start, stop);  // in place

        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, super_vertex, recvCounts, displs, MPI_TYP, MPI_COMM_WORLD);

        // out edges
        FOR(start, stop) {
            pastG[iteration + 1]->out_degree[super_vertex[parent[i]]] += pastG[iteration]->out_degree[i];  // - 2 * removed edges  || case not connected not managed ------------------------------
            if (min_edge[i] != -1) {
                pastG[iteration + 1]->out_degree[super_vertex[i]]--;
                pastG[iteration + 1]->out_degree[super_vertex[pastG[iteration]->destination[min_edge[i]]]]--;  // also possible to make a macro ma meke it more readable?
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &pastG[iteration + 1]->out_degree, next_iter_v, MPI_TYP, MPI_SUM, MPI_COMM_WORLD);

        // first edge
        mpi_exclusive_prefix_sum(pastG[iteration + 1]->out_degree, pastG[iteration + 1], start, stop);  // need new start and stop, can i already make new ones? --------------------------------

        // edges

        MPI_Wait(&req[2], MPI_STATUS_IGNORE);  // MST
        MPI_Wait(&req[3], MPI_STATUS_IGNORE);  // min_edge
        iteration++;
    }

    for (i = 1; i < iteration; ++i) free(pastG[i]);
    free(pastG);
    free(min_edge);
    free(parent);
    free(recvCounts);
    free(displs);
}

void find_min_edge(const Graph_CSR *G, TYP *min_edge, int i) {
    int first_index = G->first_edge[i];
    int min = first_index;
    for (int j = first_index + 1; j < first_index + G->out_degree; ++j) {
        if (G->weight[j] < G->weight[min]) {
            min = j;
        }
    }
    min_edge[i] = min;
}

void merge_edges(const Graph_CSR *G, TYP *MST, TYP *min_edge, TYP *parent, int i, int *removed) {
    int dest = G->destination[i];
    if (dest >= 0) {  // if not connected || unlikely()
        if (dest < i && G->destination[min_edge[dest]] == i) {
            min_edge[i] = -1;  // mirrored edge
        } else {
            parent[i] = dest;      // mfset init
            *removed++;            // removed counter
            MST[min_edge[i]] = 1;  // MST store
        }
    }
}

void parent_compression(TYP *parent, int i) {
    int father, grand_father;
    do {
        parent[i] = parent[parent[i]];  // jump liv
        father = parent[i];
        grand_father = parent[father];
    } while (father != grand_father);
}

void super_vertex_init(TYP *super_vertex, const TYP *parent, const int start, const int stop) {
    int i;

    FOR(start, stop)
    if (parent[i] != -1)  // if not connected
        super_vertex[parent[i]] = 1;
}

void mpi_exclusive_prefix_sum(int *send, int *recv, const TYP start, const TYP stop) {
    int i, local_sum = send[stop - 1], offset = 0;

    recv[0] = 0;
    FOR(start + 1, stop) recv[i] = recv[i - 1] + send[i - 1];  // Compute local prefix sum

    local_sum += recv[stop - 1];
    MPI_Exscan(&local_sum, &offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);  // get offset

    FOR(start + 1, stop) recv[i] += offset;  // apply offset
}
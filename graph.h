
#ifndef GRAPH_H
#define GRAPH_H

// adj matrix
typedef struct Graph_am {
    const int E, V;
    int** adj;
} Graph_am;

// edge list
typedef struct Edge {
    int src, dest, weight;
} Edge;

typedef struct Graph_al {
    const int E, V;
    Edge* edges;
} Graph_al;

// CSR
// duplicate edges
typedef struct Graph_CSR {
    int E;             // number of edges
    int V;             // number of vertices
    int* destination;  // array of size E, maps each edge destination its destination
    int* weight;       // array of size E, maps each edge to its weight
    int* first_edge;   // array of size V, maps each first to edge first edge
    int* out_degree;   // array of size V, maps each outdegree to the number of outgoing edges it has
} Graph_CSR;

Graph_CSR* MPI_default_graph(const int E, const int V) {
    Graph_CSR* g;
    MPI_Alloc_mem(sizeof(Graph_CSR), MPI_INFO_NULL, &g);
    if (!g) {
        perror("malloc failed");
        exit(EXIT_FAILURE);
    }
    *g = (Graph_CSR){.E = E, .V = V};
    MPI_Alloc_mem(V * sizeof(int), MPI_INFO_NULL, &g->first_edge);
    MPI_Alloc_mem(V * sizeof(int), MPI_INFO_NULL, &g->out_degree);
    for (int i = 0; i < V; ++i) g->out_degree[i] = 0;
    MPI_Alloc_mem(E * sizeof(int), MPI_INFO_NULL, &g->destination);
    MPI_Alloc_mem(E * sizeof(int), MPI_INFO_NULL, &g->weight);
    return g;
}

Graph_CSR* MPI_base_graph(const int E, const int V) {
    Graph_CSR* g;
    MPI_Alloc_mem(sizeof(Graph_CSR), MPI_INFO_NULL, &g);
    if (!g) {
        perror("malloc failed");
        exit(EXIT_FAILURE);
    }
    *g = (Graph_CSR){.E = E, .V = V, .destination = NULL, .weight = NULL, .first_edge = NULL, .out_degree = NULL};
    return g;
}

Graph_CSR* MPI_allocate_new_graph(int* out_degree, const int E, const int V) {
    Graph_CSR* g = MPI_base_graph(E, V);
    g->out_degree = out_degree;
    MPI_Alloc_mem(V * sizeof(int), MPI_INFO_NULL, &g->first_edge);
    MPI_Alloc_mem(V * sizeof(int), MPI_INFO_NULL, &g->destination);
    MPI_Alloc_mem(V * sizeof(int), MPI_INFO_NULL, &g->weight);
    return g;
}

Graph_CSR* MPI_init_graph_from_file(const char* input) {
    int E, V, v0, v1, w;
    FILE* f = fopen(input, "r");
    // g init
    fscanf(f, "%d %d", &V, &E);
    Graph_CSR* g = MPI_default_graph(E * 2, V);  // double edges
    const long bookmark = ftell(f);              // save for second iteration

    // out_degree init
    for (int e = 0; e < E; ++e) {
        fscanf(f, "%d %d %d", &v0, &v1, &w);
        g->out_degree[v0]++;
        g->out_degree[v1]++;
    }

    // first_edge init
    g->first_edge[0] = 0;
    for (int v = 1; v < V; ++v) g->first_edge[v] = g->first_edge[v - 1] + g->out_degree[v - 1];

    // edges init
    fseek(f, bookmark, SEEK_SET);
    for (int e = 0; e < E; ++e) {
        fscanf(f, "%d %d %d", &v0, &v1, &w);
        g->destination[g->first_edge[v0]] = v1;
        g->destination[g->first_edge[v1]] = v0;
        g->weight[g->first_edge[v0]] = w;
        g->weight[g->first_edge[v1]] = w;
        g->first_edge[v0]++;
        g->first_edge[v1]++;
    }

    // first_edge reset
    g->first_edge[0] = 0;
    for (int v = 1; v < V; ++v) g->first_edge[v] = g->first_edge[v - 1] + g->out_degree[v - 1];

    fclose(f);

    return g;
}

void MPI_Bcast_GraphCSR(Graph_CSR** g, MPI_Comm comm) {
    int val[2] = {0, 0};
    if (*g) {
        val[0] = (*g)->E;
        val[1] = (*g)->V;
    }
    MPI_Bcast(&val, 2, MPI_INT, 0, comm);
    if (!*g) (*g) = MPI_default_graph(val[0], val[1]);
    MPI_Request req[4];
    MPI_Ibcast((*g)->destination, val[0], MPI_INT, 0, comm, &req[0]);
    MPI_Ibcast((*g)->weight, val[0], MPI_INT, 0, comm, &req[1]);
    MPI_Ibcast((*g)->out_degree, val[1], MPI_INT, 0, comm, &req[2]);
    MPI_Ibcast((*g)->first_edge, val[1], MPI_INT, 0, comm, &req[3]);
    MPI_Waitall(4, req, MPI_STATUS_IGNORE);
}

void MPI_free_graph(Graph_CSR* g) {
    MPI_Free_mem(g->destination);
    MPI_Free_mem(g->weight);
    MPI_Free_mem(g->first_edge);
    MPI_Free_mem(g->out_degree);
    MPI_Free_mem(g);
}

#endif
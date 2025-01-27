
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
    int E, V;
    int* destination;  // array of size E, maps each edge destination its destination
    int* weight;       // array of size E, maps each edge to its weight
    int* first_edge;   // array of size V, maps each first to edge first edge
    int* out_degree;   // array of size V, maps each outdegree to the number of outgoing edges it has
} Graph_CSR;

// Graph_CSR default_graph(int E, int V) { return (Graph_CSR){.E = E, .V = V, .destination = NULL, .weight = NULL, .first_edge = NULL, .out_degree = NULL}; }

Graph_CSR* default_graph(const int E, const int V) {
    Graph_CSR* g = malloc(sizeof(Graph_CSR));
    if (!g) {
        perror("malloc failed");
        exit(EXIT_FAILURE);
    }
    *g = (Graph_CSR){.E = E, .V = V, .destination = malloc(E * sizeof(int)), .weight = malloc(E * sizeof(int)), .first_edge = malloc(V * sizeof(int)), .out_degree = calloc(V, sizeof(int))};
    return g;
}

Graph_CSR* base_graph(const int E, const int V) {
    Graph_CSR* g = malloc(sizeof(Graph_CSR));
    if (!g) {
        perror("malloc failed");
        exit(EXIT_FAILURE);
    }
    *g = (Graph_CSR){.E = E, .V = V, .destination = NULL, .weight = NULL, .first_edge = NULL, .out_degree = NULL};
    return g;
}

Graph_CSR* allocate_new_graph(int* out_degree, const int E, const int V) {  // in graph.c?
    Graph_CSR* G = base_graph(E, V);
    G->out_degree = out_degree;
    G->first_edge = malloc(V * sizeof(int));
    G->destination = malloc(E * sizeof(int));
    G->weight = malloc(E * sizeof(int));
    return G;
}

void free_graph(Graph_CSR* G) {
    free(G->destination);
    free(G->weight);
    free(G->first_edge);
    free(G->out_degree);
    free(G);
}

Graph_CSR* init_graph_from_file(const char* input) {
    int E, V, v0, v1, w;
    FILE* f = fopen(input, "r");
    // G init
    fscanf(f, "%d %d", &V, &E);
    Graph_CSR* G = default_graph(E * 2, V);  // double edges
    const long bookmark = ftell(f);          // save for second iteration

    // out_degree init
    for (int e = 0; e < E; ++e) {
        fscanf(f, "%d %d %d", &v0, &v1, &w);
        G->out_degree[v0]++;
        G->out_degree[v1]++;
    }

    // first_edge init
    G->first_edge[0] = 0;
    for (int v = 1; v < V; ++v) G->first_edge[v] = G->first_edge[v - 1] + G->out_degree[v - 1];

    // edges init
    fseek(f, bookmark, SEEK_SET);
    for (int e = 0; e < E; ++e) {
        fscanf(f, "%d %d %d", &v0, &v1, &w);
        G->destination[G->first_edge[v0]] = v1;
        G->destination[G->first_edge[v1]] = v0;
        G->weight[G->first_edge[v0]] = w;
        G->weight[G->first_edge[v1]] = w;
        G->first_edge[v0]++;
        G->first_edge[v1]++;
    }

    // first_edge reset
    G->first_edge[0] = 0;
    for (int v = 1; v < V; ++v) G->first_edge[v] = G->first_edge[v - 1] + G->out_degree[v - 1];

    fclose(f);

    return G;
}

void MPI_Bcast_GraphCSR(Graph_CSR** G, MPI_Comm comm) {
    int val[2] = {0, 0};
    if (*G) {
        val[0] = (*G)->E;
        val[1] = (*G)->V;
    }
    MPI_Bcast(&val, 2, MPI_INT, 0, comm);
    if (!*G) (*G) = default_graph(val[0], val[1]);
    MPI_Request req[4];
    MPI_Ibcast((*G)->destination, val[0], MPI_INT, 0, comm, &req[0]);
    MPI_Ibcast((*G)->weight, val[0], MPI_INT, 0, comm, &req[1]);
    MPI_Ibcast((*G)->out_degree, val[1], MPI_INT, 0, comm, &req[2]);
    MPI_Ibcast((*G)->first_edge, val[1], MPI_INT, 0, comm, &req[3]);
    MPI_Waitall(4, req, MPI_STATUS_IGNORE);
}

#endif
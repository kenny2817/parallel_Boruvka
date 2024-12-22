
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
    *g = (Graph_CSR){.E = E, .V = V, .destination = malloc(E * 2 * sizeof(int)), .weight = malloc(E * 2 * sizeof(int)), .first_edge = malloc(V * sizeof(int)), .out_degree = malloc(V * sizeof(int))};
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

void free_graph(Graph_CSR* G) {
    free(G->destination);
    free(G->weight);
    free(G->first_edge);
    free(G->out_degree);
    free(G);
}

#endif
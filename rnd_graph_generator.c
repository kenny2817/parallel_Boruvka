#include <stdio.h>
#include <stdlib.h>

#define MAX_EDGES_FOR_VERTIX(vertix) ((vertix * (vertix - 1)) / 2)

typedef struct Edge {
    long long int src, dest, weight;
} Edge;

typedef struct Graph {
    long long int V, E;
    Edge *edges;
} Graph;

void createGraph(const char *input, const int E, const int V, const int W) {
    FILE *f = fopen(input, "w");
    long long int **adj = malloc(V * sizeof(long long int *));
    if (!adj) {
        perror("Memory allocation failed");
        free(adj);
        return;
    }
    for (int v = 0, j; v < V; ++v) {
        adj[v] = calloc(V, sizeof(long long int));
        if (!adj[v]) {
            perror("Memory allocation failed");
            for (; v > 0; --v) free(adj[v]);
            free(adj);
            return;
        }
        adj[v][v] = 1;
    }
    fprintf(f, "%d %d\n", V, E);

    int rnd, rnd1, w, i = 1;
    int iter = 0;

    for (; i < V; ++i) {
        printf("\r%d", iter++);
        rnd = rand() % i;
        w = rand() % W;
        fprintf(f, "%d %d %d\n", i, rnd, w);
        adj[i][rnd] = adj[rnd][i] = 1;
    }

    for (i--; i < E; ++i) {
        printf("\r%d", iter++);
        do {
            rnd = rand() % V;
            rnd1 = rand() % V;
        } while (adj[rnd][rnd1] == 1);
        w = rand() % W;
        adj[rnd1][rnd] = adj[rnd][rnd1] = 1;
        fprintf(f, "%d %d %d\n", rnd, rnd1, w);
    }

    for (long long int i = 0; i < V; ++i) free(adj[i]);
    free(adj);
    fclose(f);
}

void createCompleteGraph(const char *input, const int V, const int W) {
    FILE *f = fopen(input, "w");
    fprintf(f, "%d %d\n", V, MAX_EDGES_FOR_VERTIX(V));

    int iter = 0;
    for (int i = 0; i < V; ++i) {
        for (int j = i + 1; j < V; ++j) {
            printf("\r%d", iter++);
            fprintf(f, "%d %d %d\n", i, j, rand() % W);
        }
    }

    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        perror("usage: rnd_graph_generator <graph.txt> <E> <V> <W>");
        return 1;
    }
    int E = atoi(argv[2]);
    int V = atoi(argv[3]);
    int W = atoi(argv[4]);

    if (E >= MAX_EDGES_FOR_VERTIX(V)) return 1;

    if (E != -1 || E < MAX_EDGES_FOR_VERTIX(V)) {
        createGraph(argv[1], E, V, W);
    } else {
        createCompleteGraph(argv[1], V, W);
    }

    return 0;
}
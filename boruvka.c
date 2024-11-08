#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT "graph.txt"
#define OUTPUT "mst.txt"

#define ITERATIONS 100
#define MAX_W 1000
#define MAX_EDGES_FOR_VERTIX(vertix) ((vertix * (vertix - 1)) / 2)
#define START_V 4
#define MAX_V 2048

typedef struct Edge {
    long long int src, dest, weight;
} Edge;

typedef struct Graph {
    long long int V, E;
    Edge *edges;
} Graph;

typedef struct MFset {
    long long int parent, rank;
} MFset;

// create a graph
void createGraph(const char *, Graph *, Graph *, long long int);

// boruvka algorithm implementation
void boruvka(Graph *, Graph *);

// find of MFset (merge-finde set, union-find set)
long long int MFind(MFset *, long long int);

// merge of MFset (merge-finde set, union-find set)
void MergeF(MFset *, long long int, long long int);

// initialize Graph and MST
void processInput(const char *, Graph *, Graph *);

// write output in file
long long int printOutput(const char *, const Graph *);

// testing boruvka alg for Iteration times
// 1 - seed (or time(null))
int main(int argc, char *argv[]) {
    srand(time(NULL));

    if (argc == 2) {
        srand(atoi(argv[1]));
    } else if (argc > 2) {
        fprintf(stderr, "Usage: %s [seed]\n", argv[0]);
        return EXIT_FAILURE;
    }

    Graph G = {0, 0, NULL};
    Graph MST = {0, 0, NULL};
    G.edges = malloc(MAX_EDGES_FOR_VERTIX(MAX_V) * sizeof(Edge));
    MST.edges = malloc((MAX_V - 1) * sizeof(Edge));
    if (!(G.edges && MST.edges)) {
        perror("Memory allocation failed");
        free(G.edges);
        free(MST.edges);
        return EXIT_FAILURE;
    }

    for (G.V = START_V; G.V <= MAX_V; G.V <<= 1) {
        // G.E = G.V + rand() % (MAX_EDGES_FOR_VERTIX(G.V) - G.V);  // rand but how can i test? rand on what level? iter or single?
        G.E = MAX_EDGES_FOR_VERTIX(G.V);  // const at max dim (complete)
        int avg_serial = 0;
        for (int i = 0, t; i < ITERATIONS; ++i) {
            printf("\r%d", i);
            fflush(stdout);
            createGraph(INPUT, &G, &MST, MAX_W);

            MST.E = 0;
            t = clock();
            boruvka(&G, &MST);
            avg_serial += clock() - t;

            // printOutput(OUTPUT, &MST);
        }
        printf("\rvertices: %10lld || time: %15.10f\n", G.V, (double)avg_serial / ITERATIONS / CLOCKS_PER_SEC);
    }

    free(MST.edges);
    free(G.edges);
    return EXIT_SUCCESS;
}

void createGraph(const char *input, Graph *G, Graph *MST, const long long int W) {
    FILE *f = fopen(input, "w");
    long long int **adj = malloc(G->V * sizeof(long long int *));
    if (!adj) {
        perror("Memory allocation failed");
        free(adj);
        return;
    }
    for (long long int v = 0, j; v < G->V; ++v) {
        adj[v] = malloc(G->V * sizeof(long long int));
        if (!adj[v]) {
            perror("Memory allocation failed");
            for (; v > 0; --v) free(adj[v]);
            free(adj);
            return;
        }
        for (j = 0; j < G->V; ++j) adj[v][j] = -1;
        adj[v][v] = 1;
    }
    fprintf(f, "%lld %lld\n", G->V, G->E);

    long long int rnd, rnd1, w, i = 1;

    for (; i < G->V; ++i) {  // parallel but critical print
        rnd = rand() % i;
        w = rand() % W;
        G->edges[i - 1] = (Edge){i, rnd, w};
        fprintf(f, "%lld %lld %lld\n", i, rnd, w);
        adj[i][rnd] = adj[rnd][i] = 1;
    }

    for (i--; i < G->E; ++i) {  // parallel but critical access on matrix
        do {
            rnd = rand() % G->V;
            rnd1 = rand() % G->V;
        } while (adj[rnd][rnd1] == 1);
        w = rand() % W;
        G->edges[i] = (Edge){rnd, rnd1, w};
        adj[rnd1][rnd] = adj[rnd][rnd1] = 1;
        fprintf(f, "%lld %lld %lld\n", rnd, rnd1, w);
    }

    for (long long int i = 0; i < G->V; ++i) free(adj[i]);
    free(adj);
    fclose(f);
}

void boruvka(Graph *G, Graph *MST) {
    MFset *set = malloc(G->V * sizeof(MFset));
    long long int *cheap = malloc(G->V * sizeof(long long int));
    if (!(set && cheap)) {
        perror("Memory allocation failed");
        free(set);
        free(cheap);
        return;
    }

    for (long long int v = 0; v < G->V; ++v) {  // parallel
        set[v].parent = v;
        set[v].rank = 0;
        cheap[v] = -1;
    }

    long long int set1, set2;
    long long int e, v, r = -1;
    while (MST->E < G->V - 1) {
        for (e = 0; e < G->E; ++e) {  // parallel with final reduce min on cheap
            set1 = MFind(set, G->edges[e].src);
            set2 = MFind(set, G->edges[e].dest);

            if (set1 != set2) {
                if (cheap[set1] == -1 || G->edges[cheap[set1]].weight > G->edges[e].weight) cheap[set1] = e;
                if (cheap[set2] == -1 || G->edges[cheap[set2]].weight > G->edges[e].weight) cheap[set2] = e;
            }
        }

        r = 0;
        for (v = 0; v < G->V; ++v) {  // parallel but merge is critical
            if (cheap[v] == -1) continue;

            set1 = MFind(set, G->edges[cheap[v]].src);
            set2 = MFind(set, G->edges[cheap[v]].dest);

            if (set1 != set2) {
                MST->edges[MST->E++] = G->edges[cheap[v]];
                MergeF(set, set1, set2);
            }
            cheap[v] = -1;
        }
    }

    free(cheap);
    free(set);
}

long long int MFind(MFset *set, long long int v) {
    if (set[v].parent != v) set[v].parent = MFind(set, set[v].parent);
    return set[v].parent;
}
void MergeF(MFset *set, long long int a, long long int b) {
    a = MFind(set, a);
    b = MFind(set, b);
    if (a != b) {
        if (set[a].rank < set[b].rank) {
            set[a].parent = b;
        } else if (set[a].rank > set[b].rank) {
            set[b].parent = a;
        } else {
            set[a].parent = b;
            set[b].rank++;
        }
    }
}

long long int printOutput(const char *output, const Graph *MST) {
    FILE *f = fopen(output, "w");
    long long int weight = 0;
    fprintf(f, "\n\nMST\n\n");
    for (long long int e = 0; e < MST->E; ++e) {
        fprintf(f, "%lld - %4lld | %4lld\n", MST->edges[e].src, MST->edges[e].dest, MST->edges[e].weight);
        weight += MST->edges[e].weight;
    }

    fprintf(f, "\nW = %5lld\n", weight);
    fclose(f);
    return weight;
}
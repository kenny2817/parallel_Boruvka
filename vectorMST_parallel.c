#include <locale.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT "graph.txt"
#define OUTPUT "mst.txt"

#define ITERATIONS 20
#define MAX_W 1000
#define MAX_EDGES_FOR_VERTIX(vertix) ((vertix * (vertix - 1)) / 2)
#define START_V 1024
#define MAX_V 1024  // 8192
#define MIN_THREADS 1
#define MAX_THREADS 20

typedef struct Edge {
    long long int src, dest, weight;
} Edge;

typedef struct Graph {
    long long int V, E;
    Edge *edges;
} Graph;

// create a graph
void createGraph(const char *, Graph *, long long int);

// boruvka algorithm implementation
void boruvka(const Graph *, long long int *);

// find of MFset (merge-finde set, union-find set)
long long int MFind(long long int *, const long long int);

// merge of MFset (merge-finde set, union-find set)
void MergeF(long long int *, long long int *, long long int, long long int);

// write output in file
long long int printOutput(const char *, const Graph *, const long long int *);
#define TIMES 6
double time_f[TIMES] = {0};

// testing boruvka alg for Iteration times
// 1 - seed (or time(null))
int main(int argc, char *argv[]) {
    setlocale(LC_NUMERIC, "fr_FR");
    srand(time(NULL));

    if (argc == 2) {
        srand(atoi(argv[1]));
    } else if (argc > 2) {
        fprintf(stderr, "Usage: %s [seed]\n", argv[0]);
        return EXIT_FAILURE;
    }

    Graph G = {0, 0, NULL};
    G.edges = malloc(MAX_EDGES_FOR_VERTIX(MAX_V) * sizeof(Edge));
    long long int *MST = malloc((MAX_V - 1) * sizeof(Edge));
    if (!(G.edges && MST)) {
        perror("Memory allocation failed");
        free(G.edges);
        free(MST);
        return EXIT_FAILURE;
    }
    FILE *f = fopen("time.txt", "a");
    FILE *x = fopen("time.csv", "a");

    for (G.V = START_V; G.V <= MAX_V; G.V <<= 1) {
        printf("\nvertices: %lld\n", G.V);
        for (int nth = MIN_THREADS; nth <= MAX_THREADS; nth += 1) {
            omp_set_num_threads(nth);
            // G.E = G.V + rand() % (MAX_EDGES_FOR_VERTIX(G.V) - G.V);  // rand but how can i test? rand on what level? iter or single?
            G.E = MAX_EDGES_FOR_VERTIX(G.V);  // const at max dim (complete)
            double t1, avg_time = 0.0;
            for (int i = 0; i < ITERATIONS; ++i) {
                printf("\r%d", i);
                fflush(stdout);
                // G.E = G.V + i * ((MAX_EDGES_FOR_VERTIX(G.V) - G.V) / (ITERATIONS - 1));  // different graph dimension per iteration (might miss golden vertices)
                createGraph(INPUT, &G, MAX_W);

                t1 = omp_get_wtime();
                boruvka(&G, MST);
                avg_time += (omp_get_wtime() - t1);  // normalized i think / (i ? i : 1)

                // printOutput(OUTPUT, &G, &MST);
            }
            fprintf(x, "%d.%lld", nth, G.V);
            fprintf(f, "nth %3d || vertices %10lld", nth, G.V);
            for (int i = 0; i < TIMES; ++i) {
                fprintf(x, ".%f", time_f[i] / ITERATIONS);
                fprintf(f, " || f%d %15.10f", i, time_f[i] / ITERATIONS);
                time_f[i] = 0;
            }
            fprintf(f, "\n");
            fprintf(x, "\n");
            printf("\rnth: %3d || omptime: %15.10f\n", omp_get_max_threads(), avg_time / ITERATIONS);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "===================================================================================================================================================================\n\n");

    fclose(x);
    fclose(f);
    free(MST);
    free(G.edges);
    return EXIT_SUCCESS;
}

void createGraph(const char *input, Graph *G, const long long int W) {
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

void boruvka(const Graph *G, long long int *MST) {
    double t, t1;
    t = omp_get_wtime();
    long long int parent1, parent2, cheap1, cheap2;
    long long int e, v, e_left = G->V - 1;
    const long long int size = G->V * sizeof(long long int);
    long long int *parent = malloc(size);
    long long int *rank = malloc(size);
    long long int *cheap = malloc(size);
    omp_lock_t *lock = malloc(G->V * sizeof(omp_lock_t));
    if (!parent || !rank || !cheap || !lock) {
        perror("Memory allocation failed");
        free(parent);
        free(rank);
        free(cheap);
        free(lock);
        return;
    }
    time_f[0] += omp_get_wtime() - t;
    t = omp_get_wtime();
    // dividing the MFset there is no race conditions
    // #pragma omp parallel for default(none) shared(G, parent, rank, cheap, lock)
    for (long long int v = 0; v < G->V; ++v) {
        parent[v] = v;
        rank[v] = 0;
        cheap[v] = -1;
        omp_init_lock(&lock[v]);
    }
    time_f[1] += omp_get_wtime() - t;

    t = omp_get_wtime();
    while (e_left > 0) {
        // number of threads proportional to the if probability using e_left?
        // shared the critical part would make serial based on probability
        // reduce on cheap O(v/PlogP)
        t1 = omp_get_wtime();
#pragma omp parallel for default(none) shared(G, parent, cheap, lock) private(parent1, parent2, cheap1, cheap2) schedule(dynamic)
        for (e = 0; e < G->E; ++e) {
            parent1 = parent[G->edges[e].src];
            parent2 = parent[G->edges[e].dest];

            if (parent1 != parent2) {
                omp_set_lock(&lock[parent1]);
                cheap1 = cheap[parent1];
                if (cheap1 == -1 || G->edges[cheap1].weight > G->edges[e].weight) cheap[parent1] = e;
                omp_unset_lock(&lock[parent1]);

                omp_set_lock(&lock[parent2]);
                cheap2 = cheap[parent2];
                if (cheap2 == -1 || G->edges[cheap2].weight > G->edges[e].weight) cheap[parent2] = e;
                omp_unset_lock(&lock[parent2]);
            }
        }
        time_f[3] += omp_get_wtime() - t1;

        t1 = omp_get_wtime();
        long long int c;
        for (v = 0; v < G->V; ++v) {
            c = cheap[v];
            if (c != -1) {
                parent1 = MFind(parent, G->edges[c].src);
                parent2 = MFind(parent, G->edges[c].dest);

                if (parent1 != parent2) {
                    MST[--e_left] = c;
                    MergeF(parent, rank, parent1, parent2);
                }
                cheap[v] = -1;
            }
        }
        time_f[4] += omp_get_wtime() - t1;

        t1 = omp_get_wtime();
        for (v = 0; v < G->V; ++v) parent1 = MFind(parent, v);
        time_f[5] += omp_get_wtime() - t1;
    }
    time_f[2] += omp_get_wtime() - t;

    free(cheap);
    free(parent);
    free(rank);
    for (int v = 0; v < G->V; ++v) omp_destroy_lock(&lock[v]);
    free(lock);
}

// long long int MFind(long long int *parent, long long int v) {
//     if (parent[v] != v) parent[v] = MFind(parent, parent[v]);
//     return parent[v];
// }
long long int MFind(long long int *parent, const long long int v) {
    long long int par;
#pragma omp atomic read
    par = parent[v];
    if (par != v) {
        par = MFind(parent, par);
#pragma omp atomic write
        parent[v] = par;
    }
    return par;
}

void MergeF(long long int *parent, long long int *rank, long long int a, long long int b) {
    a = MFind(parent, a);
    b = MFind(parent, b);
    if (a != b) {
        if (rank[a] < rank[b]) {
            parent[a] = b;
        } else if (rank[a] > rank[b]) {
            parent[b] = a;
        } else {
            parent[a] = b;
            rank[b]++;
        }
    }
}

long long int printOutput(const char *output, const Graph *G, const long long int *MST) {
    FILE *f = fopen(output, "w");
    long long int weight = 0;
    fprintf(f, "\n\nMST\n\n");
    for (long long int e = 0; e < G->V - 1; ++e) {
        fprintf(f, "%lld - %4lld | %4lld\n", G->edges[MST[e]].src, G->edges[MST[e]].dest, G->edges[MST[e]].weight);
        weight += G->edges[MST[e]].weight;
    }

    fprintf(f, "\nW = %5lld\n", weight);
    fclose(f);
    return weight;
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT "graph.txt"
#define OUTPUT "mst.txt"
#define VISUAL "graph.dot"

typedef struct Edge {
    int src, dest, weight;
} Edge;

typedef struct Graph {
    int V, E;
    Edge *edges;
} Graph;

typedef struct MFset {
    int parent, rank;
} MFset;

// create a graph
void createGraph(const char *, Graph *, Graph *, int);

// boruvka algorithm implementation + edge optimization
//
// edge cutting optimization:
// right now boruvka consumes G creating MST because of the optimization
// it is possible to save E and to swap instead of just copy
void boruvka(Graph *, Graph *);

// find of MFset (merge-finde set, union-find set)
int MFind(MFset *, int);

// merge of MFset (merge-finde set, union-find set)
void MergeF(MFset *, int, int);

// initialize Graph and MST
void processInput(const char *, Graph *, Graph *);

// write output in file
void printOutput(const char *, const Graph *);

// make file for visualization graphViz
void graphVis(const char *, const Graph *, const Graph *);

int main() {
    Graph G = {0, 0, NULL};
    Graph MST = {0, 0, NULL};

    char input_adress[100];
    int file_random;
    printf("Graph from file [y/N] : ");
    scanf("%c", &file_random);

    if (file_random != 'y' || file_random != 'Y') {
        printf("Vertices ad Edges, max Weight:");
        int W;
        scanf("%d %d %d", &G.V, &G.E, &W);
        if (G.E + 1 < G.V || G.E * 2 > G.V * G.V) return EXIT_FAILURE;
        createGraph(INPUT, &G, &MST, W);
    } else {
        printf("file name: ");
        scanf("%s", input_adress);
        processInput(input_adress, &G, &MST);
        if (G.E + 1 < G.V) return EXIT_FAILURE;
    }

    boruvka(&G, &MST);

    printOutput(OUTPUT, &MST);
    graphVis(VISUAL, &G, &MST);

    free(G.edges);
    free(MST.edges);
    return EXIT_SUCCESS;
}

void createGraph(const char *input, Graph *G, Graph *MST, const int W) {
    srand(time(NULL));
    FILE *f = fopen(input, "w");
    fprintf(f, "%d %d\n", G->V, G->E);
    G->edges = malloc(G->E * sizeof(Edge));
    MFset *set = malloc(G->V * sizeof(MFset));
    int **adj = malloc(G->V * sizeof(int *));

    for (int i = 0; i < G->V; ++i) {
        adj[i] = malloc(G->V * sizeof(int));
        for (int j = 0; j < G->V; ++j) adj[i][j] = -1;
        adj[i][i] = 1;
        set[i].parent = i;
        set[i].rank = 0;
    }

    int set1, set2;
    int rnd, rnd1, w, i = 0;

    for (; i < G->V - 1; ++i) {
        do {
            rnd = rand() % G->V;
            set1 = MFind(set, i);
            set2 = MFind(set, rnd);
        } while (set1 == set2);

        w = rand() % W;
        G->edges[i] = (Edge){i, rnd, w};
        fprintf(f, "%d %d %d\n", i, rnd, w);
        adj[i][rnd] = 1;
        adj[rnd][i] = 1;
        MergeF(set, set1, set2);
    }
    free(set);

    for (; i < G->E; ++i) {
        do {
            rnd = rand() % G->V;
            rnd1 = rand() % G->V;
        } while (adj[rnd][rnd1] == 1);
        w = rand() % W;
        G->edges[i] = (Edge){rnd, rnd1, w};
        adj[rnd1][rnd] = 1;
        adj[rnd][rnd1] = 1;
        fprintf(f, "%d %d %d\n", rnd, rnd1, w);
    }

    fclose(f);
    for (int i = 0; i < G->V; ++i) free(adj[i]);
    free(adj);
    MST->edges = malloc((G->V - 1) * sizeof(Edge));
}

void boruvka(Graph *G, Graph *MST) {
    MFset *set = malloc(G->V * sizeof(MFset));
    int *cheap = malloc(G->V * sizeof(int));
    int *removable = malloc(G->V * sizeof(int));

    for (int v = 0; v < G->V; ++v) {
        set[v].parent = v;
        set[v].rank = 0;
        cheap[v] = -1;
        removable[v] = -1;
    }

    int set1, set2;
    int e, v, r = -1;
    while (MST->E < G->V - 1) {
        for (e = 0; e < G->E; ++e) {
            set1 = MFind(set, G->edges[e].src);
            set2 = MFind(set, G->edges[e].dest);

            if (set1 == set2) {
                // G->edges[e--] = G->edges[--(G->E)];  // edge cutting optimization off because doesnt help for visualization
            } else {
                if (cheap[set1] == -1 || G->edges[cheap[set1]].weight > G->edges[e].weight) cheap[set1] = e;
                if (cheap[set2] == -1 || G->edges[cheap[set2]].weight > G->edges[e].weight) cheap[set2] = e;
            }
        }

        for (v = 0; v < G->V; ++v) {
            if (cheap[v] == -1) continue;

            set1 = MFind(set, G->edges[cheap[v]].src);
            set2 = MFind(set, G->edges[cheap[v]].dest);

            if (set1 != set2) {
                MST->edges[MST->E++] = G->edges[cheap[v]];
                MergeF(set, set1, set2);
                removable[++r] = cheap[v];
            }
            cheap[v] = -1;
        }
        for (; r >= 0; --r) G->edges[removable[r]] = G->edges[--(G->E)];  // edge cutting optimization
    }
    free(removable);
    free(cheap);
    free(set);
}

int MFind(MFset *set, int n) {
    if (set[n].parent != n) set[n].parent = MFind(set, set[n].parent);
    return set[n].parent;
}

void MergeF(MFset *set, int a, int b) {
    a = MFind(set, a);
    b = MFind(set, b);
    if (a == b) return;
    if (set[a].rank < set[b].rank) {
        set[a].parent = b;
    } else if (set[a].rank > set[b].rank) {
        set[b].parent = a;
    } else {
        set[a].parent = b;
        set[b].rank++;
    }
}

void processInput(const char *input, Graph *G, Graph *MST) {
    printf("%s\n", input);
    FILE *f = fopen(input, "r");
    fscanf(f, "%d %d", &G->V, &G->E);
    G->edges = malloc(G->E * sizeof(Edge));
    for (int e = 0; e < G->E; ++e) {
        fscanf(f, "%d %d %d", &G->edges[e].src, &G->edges[e].dest, &G->edges[e].weight);
    }
    fclose(f);
    MST->edges = malloc((G->V - 1) * sizeof(Edge));
}

void printOutput(const char *output, const Graph *MST) {
    FILE *f = fopen(output, "w");
    int weight = 0;
    fprintf(f, "\n\nMST\n\n");
    for (int e = 0; e < MST->E; weight += MST->edges[e++].weight) {
        fprintf(f, "%4d - %4d | %4d\n", MST->edges[e].src, MST->edges[e].dest, MST->edges[e].weight);
    }
    fprintf(f, "\nW = %5d\n", weight);
    fclose(f);
}

#define G_COLOR "black"
#define G_WIDTH 1
#define MST_COLOR "red"
#define MST_WIDTH 2

void graphVis(const char *visual, const Graph *G, const Graph *MST) {
    FILE *f = fopen(visual, "w");
    fprintf(f, "graph G {\n");
    fprintf(f, "\tnode [shape=circle];\n");
    for (int i = 0; i < G->E; i++) {
        fprintf(f, "\t%d -- %d [label=%d, color=%s, penwidth=%d];\n", G->edges[i].src, G->edges[i].dest, G->edges[i].weight, G_COLOR, G_WIDTH);
    }
    for (int i = 0; i < MST->E; i++) {
        fprintf(f, "\t%d -- %d [label=%d, color=%s, penwidth=%d];\n", MST->edges[i].src, MST->edges[i].dest, MST->edges[i].weight, MST_COLOR, MST_WIDTH);
    }
    fprintf(f, "}\n");
    fclose(f);
}
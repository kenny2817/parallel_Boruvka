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
void createGraph(const char *, Graph *, Graph *);

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

// make file for visualization
void graphVis(const char *, const Graph *, const Graph *);

int main() {
    Graph G = {0, 0, NULL};
    Graph MST = {0, 0, NULL};

    char *input_adress;
    int file_random;
    printf("input Graph:\n - 0 file\n - 1 random\n");
    scanf("%d", &file_random);

    if (file_random) {
        printf("Vertices ad Edges:");
        scanf("%d %d", &G.V, &G.E);
        if (G.E + 1 < G.V) return EXIT_FAILURE;
        createGraph(INPUT, &G, &MST);
    } else {
        printf("file name: ");
        scanf("%s", &input_adress);
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

void createGraph(const char *input, Graph *G, Graph *MST) {
    srand(time(NULL));
    FILE *f = fopen(input, "a");
    G->edges = malloc(G->E * sizeof(Edge));
    int *randomizedAccess = malloc(G->V * sizeof(int));
    MFset *set = malloc(G->V * sizeof(MFset));
    for (int v = 0; v < G->V; ++v) {
        set[v].parent = v;
        set[v].rank = 0;
        randomizedAccess[v] = v;
    }
    for (int v = 0, index; v < G->V; ++v) {
        index = rand() % G->V;
        randomizedAccess[v] = index;
        randomizedAccess[index] = v;
    }

    for (int v = 1; v < G->V; ++v) {
        // creo edge se colleghi 2 set differenti e merge
        }

    free(randomizedAccess);
    free(set);
    fclose(f);
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
    FILE *f = fopen(output, "a");
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
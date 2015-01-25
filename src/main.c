#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

const int EDGE_MEMBERS = 3;
const int UNSET_CANONICAL_ELEMENT = -1;
const int NO_EDGE = -1;
const int MAXIMUM_RANDOM = 20;

typedef struct Handle {
	int algorithm;
	int columns;
	int help;
	int maze;
	int new;
	int parallel;
	int rows;
	int verbose;
} Handle;

typedef struct ListElement {
	int vertex;
	int weight;
} ListElement;

typedef struct List {
	int alloced;
	int size;
	ListElement* elements;
} List;

typedef struct AdjacencyList {
	int elements;
	List* lists;
} AdjacencyList;

typedef struct Set {
	int elements;
	int* canonicalElements;
	int* rank;
} Set;

typedef struct WeightedGraph {
	int edges;
	int vertices;
	int* edgeList;
} WeightedGraph;

typedef struct BinaryHeapElement {
	int vertex;
	int via;
	int weight;
} BinaryHeapElement;

typedef struct BinaryMinHeap {
	int alloced;
	int size;
	int* positions;
	BinaryHeapElement* elements;
} BinaryMinHeap;

typedef struct FibonacciHeapElement {
	int vertex;
	int via;
	int weight;
	int marked;
	int childrens;
	struct FibonacciHeapElement* parent;
	struct FibonacciHeapElement* child;
	struct FibonacciHeapElement* left;
	struct FibonacciHeapElement* right;
} FibonacciHeapElement;

typedef struct FibonacciMinHeap {
	int size;
	FibonacciHeapElement** positions;
	FibonacciHeapElement* minimum;
} FibonacciMinHeap;

void consolidateHeap(FibonacciMinHeap* heap);
void createMazeFile(const int rows, const int columns,
		const char outputFileName[]);
int createsLoop(const WeightedGraph* graph, int currentedge, Set* set);
void cutFibonacciMinHeap(FibonacciMinHeap* heap, FibonacciHeapElement* element);
void decreaseBinaryMinHeap(BinaryMinHeap* heap, const int vertex, const int via,
		const int weight);
void decreaseFibonacciMinHeap(FibonacciMinHeap* heap, const int vertex,
		const int via, const int weight);
void deleteAdjacencyList(AdjacencyList* list);
void deleteBinaryMinHeap(BinaryMinHeap* heap);
void deleteFibonacciMinHeap(FibonacciMinHeap* heap);
void deleteSet(Set* set);
void deleteWeightedGraph(WeightedGraph* graph);
int findSet(Set* set, int vertex);
void heapifyBinaryMinHeap(BinaryMinHeap* heap, int position);
void heapifyDownBinaryMinHeap(BinaryMinHeap* heap, int position);
void insertFibonacciMinHeap(FibonacciMinHeap* heap,
		FibonacciHeapElement* element);
void merge(int* edgeList, int start, int size, int pivot);
void mergeSort(int* edgeList, int start2, int size2);
void mstBoruvka(const WeightedGraph* graph, const WeightedGraph* mst);
void mstKruskal(WeightedGraph* graph, const WeightedGraph* mst);
void mstPrimBinary(const WeightedGraph* graph, const WeightedGraph* mst);
void mstPrimFibonacci(const WeightedGraph* graph, const WeightedGraph* mst);
void newAdjacencyList(AdjacencyList* list, const WeightedGraph* graph);
void newBinaryMinHeap(BinaryMinHeap* heap);
void newFibonacciMinHeap(FibonacciMinHeap* heap);
void newSet(Set* set, const int elements);
void newWeightedGraph(WeightedGraph* graph, const int vertices, const int edges);
void popBinaryMinHeap(BinaryMinHeap* heap, int* vertex, int* via, int* weight);
void newFibonacciHeapElement(FibonacciHeapElement* element, int vertex, int via,
		int weight, FibonacciHeapElement* left, FibonacciHeapElement* right,
		FibonacciHeapElement* parent, FibonacciHeapElement* child);
void popFibonacciMinHeap(FibonacciMinHeap* heap, int* vertex, int* via,
		int* weight);
void printAdjacencyList(const AdjacencyList* list);
void printBinaryHeap(const BinaryMinHeap* heap);
void printFibonacciHeap(const FibonacciMinHeap* heap,
		FibonacciHeapElement* startElement);
void printMaze(const WeightedGraph* graph, int rows, int columns);
void printSet(const Set* set);
void printWeightedGraph(const WeightedGraph* graph);
Handle processParameters(int argc, char* argv[]);
void pushAdjacencyList(AdjacencyList* list, int from, int to, int weight);
void pushBinaryMinHeap(BinaryMinHeap* heap, const int vertex, const int via,
		const int weight);
void pushFibonacciMinHeap(FibonacciMinHeap* heap, const int vertex,
		const int via, const int weight);
void readMazeFile(WeightedGraph* graph, const char inputFileName[]);
void sort(WeightedGraph* graph);
void swapBinaryHeapElement(BinaryMinHeap* heap, int position1, int position2);
FibonacciHeapElement* updateDegreeFibonacciMinHeap(
		FibonacciHeapElement** degree, FibonacciHeapElement* parent,
		FibonacciHeapElement* child);
void unionSet(Set* set, const int parent1, const int parent2);

/*
 * main program
 */
int main(int argc, char* argv[]) {
	// MPI variables and initialization
	int size;
	int rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Datatype MPI_HANDLE;
	MPI_Type_contiguous(sizeof(Handle) / sizeof(int), MPI_INT, &MPI_HANDLE);
	MPI_Type_commit(&MPI_HANDLE);

	// control variable
	Handle handle;

	// graph Variables
	WeightedGraph* graph = &(WeightedGraph ) { .edges = 0, .vertices = 0,
					.edgeList = NULL };
	WeightedGraph* mst = &(WeightedGraph ) { .edges = 0, .vertices = 0,
					.edgeList = NULL };

	if (rank == 0) {
		printf("Starting\n");

		// process command line parameters
		handle = processParameters(argc, argv);
	}

	MPI_Bcast(&handle, 1, MPI_HANDLE, 0, MPI_COMM_WORLD);
	if (handle.help == 1) {
		MPI_Finalize();
		exit(0);
	}

	if (rank == 0) {
		if (handle.new == 1) {
			// create a new maze file
			createMazeFile(handle.rows, handle.columns, "maze.csv");
		}

		// read the maze file and store it in the graph
		readMazeFile(graph, "maze.csv");

		if (handle.verbose == 1) {
			// print the edges of the read graph
			printf("Graph:\n");
			printWeightedGraph(graph);
		}

		newWeightedGraph(mst, graph->vertices, graph->vertices - 1);
	}

	double start = MPI_Wtime();
	if (handle.algorithm == 0) {
		// use Kruskal's algorithm
		mstKruskal(graph, mst);
	} else if (handle.algorithm == 1) {
		// use Prim's algorithm (fibonacci)
		mstPrimFibonacci(graph, mst);
	} else if (handle.algorithm == 2) {
		// use Prim's algorithm (binary)
		mstPrimBinary(graph, mst);
	} else if (handle.algorithm == 3) {
		// use Boruvka's algorithm
		mstBoruvka(graph, mst);
	}

	if (rank == 0) {
		printf("Time elapsed: %f s\n", MPI_Wtime() - start);

		if (handle.verbose == 1) {
			// print the edges of the MST
			printf("MST:\n");
			printWeightedGraph(mst);
		}

		int mstWeight = 0;
		for (int i = 0; i < mst->edges; i++) {
			mstWeight += mst->edgeList[i * EDGE_MEMBERS + 2];
		}
		printf("MST weight: %d\n", mstWeight);

		if (handle.maze == 1) {
			// print the maze to the console
			printf("Maze:\n");
			printMaze(mst, handle.rows, handle.columns);
		}

		// cleanup
		deleteWeightedGraph(graph);
		deleteWeightedGraph(mst);

		printf("Finished\n");
	}

	MPI_Finalize();

	return 0;
}

/*
 * TODO check
 * rearrange fibonacci heap and update minimum
 */
void consolidateHeap(FibonacciMinHeap* heap) {
	// initialize degree array
	int degreeSize = 2 * log2(heap->size) + 1;
	FibonacciHeapElement** degree = (FibonacciHeapElement**) malloc(
			degreeSize * sizeof(FibonacciHeapElement*));
	for (int i = 0; i < degreeSize; i++) {
		degree[i] = NULL;
	}

	// link trees with same degree together
	FibonacciHeapElement* startElement = heap->minimum;
	FibonacciHeapElement* currentElement = startElement;
	int currentDegree = 0;
	do {
		currentDegree = currentElement->childrens;
		if (degree[currentDegree] == NULL) {
			degree[currentDegree] = currentElement;
		} else {
			if (degree[currentDegree]->weight > currentElement->weight) {
				// insert degree[currentDegree] under currentElement
				currentElement = updateDegreeFibonacciMinHeap(degree,
						currentElement, degree[currentDegree]);
			} else {
				// insert currentElement under degree[currentDegree]
				currentElement = updateDegreeFibonacciMinHeap(degree,
						degree[currentDegree], currentElement);
			}
		}
		currentElement = currentElement->right;
		for (int i = 0; i < degreeSize; i++) {
			printf("%p ", degree[i]);
		}
		printf("\n");
	} while (currentElement != startElement);

	for (int i = 0; i < degreeSize; i++) {
		printf("%p ", degree[i]);
	}
	printf("\n");

	// update minimum
	heap->minimum = NULL;
	for (int i = 0; i < degreeSize; i++) {
		if (degree[i] != NULL) {
			if (heap->minimum == NULL) {
				heap->minimum = degree[i];
			} else {
				if (degree[i]->weight < heap->minimum->weight) {
					heap->minimum = degree[i];
				}
			}
		}
	}
	free(degree);
}

/*
 * save a 2D (rows x columns) grid graph with random edge weights to a file
 */
void createMazeFile(const int rows, const int columns,
		const char outputFileName[]) {
	// open the file
	FILE* outputFile;
	const char* outputMode = "w";
	outputFile = fopen(outputFileName, outputMode);
	if (outputFile == NULL) {
		printf("Could not open output file, exiting!\n");
		exit(1);
	}

	// first line contains number of vertices and edges
	const int vertices = rows * columns;
	const int edges = vertices * 2 - rows - columns;
	fprintf(outputFile, "%d %d\n", vertices, edges);

	// all lines after the first contain the edges, values stored as "from to weight"
	srand(time(NULL));
	int vertex;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			vertex = i * columns + j;
			if (j != columns - 1) {
				fprintf(outputFile, "%d %d %d\n", vertex, vertex + 1,
						rand() % MAXIMUM_RANDOM);
			}
			if (i != rows - 1) {
				fprintf(outputFile, "%d %d %d\n", vertex, vertex + columns,
						rand() % MAXIMUM_RANDOM);
			}
		}
	}

	fclose(outputFile);
}

/*
 * check if adding the edge to the MST would create a loop
 */
int createsLoop(const WeightedGraph* graph, int currentEdge, Set* set) {
	int from = graph->edgeList[currentEdge * EDGE_MEMBERS];
	int to = graph->edgeList[currentEdge * EDGE_MEMBERS + 1];
	if (set->canonicalElements[from] == set->canonicalElements[to]) {
		// adding the edge would create a loop
		return 1;
	} else {
		// adding the edge wouldn't create a loop, update canonical elements
		int replace = set->canonicalElements[from];
		for (int i = 0; i < graph->vertices; i++) {
			if (set->canonicalElements[i] == replace) {
				set->canonicalElements[i] = set->canonicalElements[to];
			}
		}
		return 0;
	}
}

/*
 * cut an element from a fibonacci heap
 */
void cutFibonacciMinHeap(FibonacciMinHeap* heap, FibonacciHeapElement* element) {
	FibonacciHeapElement* parent = element->parent;
	if (parent != NULL) {
		parent->childrens--;
	}
	if (element->right == element) {
		// only one element in the child list
		parent->child = NULL;
	} else {
		element->right->left = element->left;
		element->left->right = element->right;
		if (parent->child == element) {
			// update parents child pointer
			parent->child = element->right;
		}
	}

	// insert as new root element
	insertFibonacciMinHeap(heap, element);
	element->parent = NULL;

	if (parent->parent != NULL) {
		// not a root element
		if (parent->marked) {
			// recursively cut marked parent
			cutFibonacciMinHeap(heap, parent);
			parent->marked = 0;
		} else {
			parent->marked = 1;
		}
	}
}

/*
 * cleanup adjacency list data
 */
void deleteAdjacencyList(AdjacencyList* list) {
	for (int i = 0; i < list->elements; i++) {
		free(list->lists[i].elements);
	}
	free(list->lists);
}

/*
 * only decrease the weight to a given vertex
 */
void decreaseBinaryMinHeap(BinaryMinHeap* heap, const int vertex, const int via,
		const int weight) {
	if (heap->positions[vertex] != -1
			&& (heap->elements[heap->positions[vertex]].weight > weight)) {
		heap->elements[heap->positions[vertex]].via = via;
		heap->elements[heap->positions[vertex]].weight = weight;
		heapifyBinaryMinHeap(heap, heap->positions[vertex]);
	}
}

/*
 * only decrease the weight to a given vertex
 */
void decreaseFibonacciMinHeap(FibonacciMinHeap* heap, const int vertex,
		const int via, const int weight) {
	FibonacciHeapElement* element = heap->positions[vertex];
	if (element != NULL && (element->weight > weight)) {
		element->via = via;
		element->weight = weight;
		if (element->parent == NULL) {
			if (element->weight < heap->minimum->weight) {
				heap->minimum = element;
			}
		} else if (weight < element->parent->weight) {
			// if heap property is violated cut off the element
			cutFibonacciMinHeap(heap, element);
		}
	}
}

/*
 * cleanup binary heap data
 */
void deleteBinaryMinHeap(BinaryMinHeap* heap) {
	free(heap->elements);
}

/*
 * cleanup fibonacci heap data
 */
void deleteFibonacciMinHeap(FibonacciMinHeap* heap) {
	free(heap->positions);
}

/*
 * cleanup set data
 */
void deleteSet(Set* set) {
	free(set->canonicalElements);
	free(set->rank);
}

/*
 * cleanup graph data
 */
void deleteWeightedGraph(WeightedGraph* graph) {
	free(graph->edgeList);
}

/*
 * return the canonical element of a vertex with path compression
 */
int findSet(Set* set, int vertex) {
	if (set->canonicalElements[vertex] == UNSET_CANONICAL_ELEMENT) {
		return vertex;
	} else {
		set->canonicalElements[vertex] = findSet(set,
				set->canonicalElements[vertex]);
		return set->canonicalElements[vertex];
	}
}

/*
 * check and restore heap property from given position upwards
 */
void heapifyBinaryMinHeap(BinaryMinHeap* heap, int position) {
	while (position >= 0) {
		int positionParent = (position - 1) / 2;
		if (heap->elements[position].weight
				< heap->elements[positionParent].weight) {
			swapBinaryHeapElement(heap, position, positionParent);
			position = positionParent;
		} else {
			break;
		}
	}
}

/*
 * check and restore heap property from given position upwards
 */
void heapifyDownBinaryMinHeap(BinaryMinHeap* heap, int position) {
	while (position < heap->size) {
		int positionLeft = (position + 1) * 2 - 1;
		int positionRight = (position + 1) * 2;
		int positionSmallest = position;
		if ((positionLeft <= heap->size)
				&& (heap->elements[positionLeft].weight
						< heap->elements[positionSmallest].weight)) {
			positionSmallest = positionLeft;
		}
		if ((positionRight <= heap->size)
				&& (heap->elements[positionRight].weight
						< heap->elements[positionSmallest].weight)) {
			positionSmallest = positionRight;
		}
		if (heap->elements[position].weight
				> heap->elements[positionSmallest].weight) {
			swapBinaryHeapElement(heap, position, positionSmallest);
			position = positionSmallest;
		} else {
			break;
		}
	}
}

/*
 * merge element into fibonacci heap left to the minimum
 */
void insertFibonacciMinHeap(FibonacciMinHeap* heap,
		FibonacciHeapElement* element) {
	if (heap->minimum == NULL) {
		heap->minimum = element;
	} else {
		FibonacciHeapElement* endHeap = heap->minimum->left;
		heap->minimum->left = element;
		element->left = endHeap;
		endHeap->right = element;
		element->right = heap->minimum;

		// set new minimum
		if (heap->minimum->weight > element->weight) {
			heap->minimum = element;
		}
	}
}

/*
 * merge sorted lists
 */
void merge(int* edgeList, int start, int size, int pivot) {
//	int length = size - start + 1;
//	int* working = (int*) malloc(length * EDGE_MEMBERS * sizeof(int));
//	memset(working, 0, length * EDGE_MEMBERS * sizeof(int));
//	for (int i = 0; i <= pivot; i++) {
//		for (int j = 0; j < EDGE_MEMBERS; j++) {
//			working[i * EDGE_MEMBERS + j] = edgeList[(i + start) * EDGE_MEMBERS
//					+ j];
//		}
//	}
//	for (int i = pivot + 1; i < length; i++) {
//		for (int j = 0; j < EDGE_MEMBERS; j++) {
//			working[(length - i) * EDGE_MEMBERS + j] = edgeList[(i + start)
//					* EDGE_MEMBERS + j];
//		}
//	}

//	for (i = pivot + 1; i > start; i--) {
//		working[i - 1] = edgeList[i - 1];
//	}
//	for (j = pivot; j < size; j++) {
//		working[size + pivot - j] = edgeList[j + 1];
//	}

//	int i = 0;
//	int j = size;
//	for (int k = start; k <= size; k++) {
//		if (working[j * EDGE_MEMBERS + 2] < working[i * EDGE_MEMBERS + 2]) {
//			for (int l = 0; l < EDGE_MEMBERS; l++) {
//				edgeList[k * EDGE_MEMBERS + l] = working[i * EDGE_MEMBERS + l];
//			}
//			j--;
//		} else {
//			for (int l = 0; l < EDGE_MEMBERS; l++) {
//				edgeList[k * EDGE_MEMBERS + l] = working[i * EDGE_MEMBERS + l];
//			}
//			i++;
//		}
//	}
//	free(working);

	int length = size - start + 1;
	// make a temporary copy of the list for merging
	int* working = (int*) malloc(length * EDGE_MEMBERS * sizeof(int));
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < EDGE_MEMBERS; j++) {
			working[i * EDGE_MEMBERS + j] = edgeList[(i + start) * EDGE_MEMBERS
					+ j];
		}
	}

	// merge the two parts together
	int merge1 = 0;
	int merge2 = pivot - start + 1;
	for (int i = 0; i < length; i++) {
		if (merge2 <= size - start) {
			if (merge1 <= pivot - start) {
				if (working[merge1 * EDGE_MEMBERS + 2]
						> working[merge2 * EDGE_MEMBERS + 2]) {
					for (int j = 0; j < EDGE_MEMBERS; j++) {
						edgeList[(i + start) * EDGE_MEMBERS + j] =
								working[merge2 * EDGE_MEMBERS + j];
					}
					merge2++;
				} else {
					for (int j = 0; j < EDGE_MEMBERS; j++) {
						edgeList[(i + start) * EDGE_MEMBERS + j] =
								working[merge1 * EDGE_MEMBERS + j];
					}
					merge1++;
				}
			} else {
				for (int j = 0; j < EDGE_MEMBERS; j++) {
					edgeList[(i + start) * EDGE_MEMBERS + j] = working[merge2
							* EDGE_MEMBERS + j];
				}
				merge2++;
			}
		} else {
			for (int j = 0; j < EDGE_MEMBERS; j++) {
				edgeList[(i + start) * EDGE_MEMBERS + j] = working[merge1
						* EDGE_MEMBERS + j];
			}
			merge1++;
		}
	}

	free(working);
}

/*
 * sort the edge list using merge sort
 */
void mergeSort(int* edgeList, int start, int size) {
	if (start == size) {
		// already sorted
		return;
	}

	// recursively divide the list in two parts and sort them
	int pivot = (start + size) / 2;
	mergeSort(edgeList, start, pivot);
	mergeSort(edgeList, pivot + 1, size);

	merge(edgeList, start, size, pivot);
}

/*
 * TODO
 *
 * find a MST of the graph using Boruvka's algorithm
 */
void mstBoruvka(const WeightedGraph* graph, const WeightedGraph* mst) {
	printf("DUMMY!\n");

	// create needed data structures
	Set* set = &(Set ) { .elements = 0, .canonicalElements = NULL, .rank =
			NULL };
	//AdjacencyList* list = &(AdjacencyList ) { .elements = 0, .lists = NULL };
	newSet(set, graph->vertices);
	//newAdjacencyList(list, graph);

	// clean up
	deleteSet(set);

	/*		// foreach tree in forest, find closest edge
	 // if edge weights are equal, ties are broken in favor of first edge
	 // in G.Edge()
	 Edge[] closest = new Edge[G.V()];
	 for (Edge e : G.Edge())
	 {
	 int v = e.either(), w = e.other(v);
	 int i = uf.find(v), j = uf.find(w);
	 if (i == j)
	 continue;   // same tree
	 if (closest[i] == null || less(e, closest[i]))
	 closest[i] = e;
	 if (closest[j] == null || less(e, closest[j]))
	 closest[j] = e;
	 }

	 // add newly discovered Edge to MST
	 for (int i = 0; i < G.V(); i++) {
	 Edge e = closest[i];
	 if (e != null) {
	 int v = e.either(), w = e.other(v);
	 // don't add the same edge twice
	 if (!uf.connected(v, w)) {
	 mst.add(e);
	 weight += e.weight();
	 uf.union(v, w);
	 }
	 }*/
}

/*
 * find a MST of the graph using Kruskal's algorithm
 */
void mstKruskal(WeightedGraph* graph, const WeightedGraph* mst) {
	// create needed data structures
	Set* set = &(Set ) { .elements = 0, .canonicalElements = NULL, .rank =
			NULL };
	newSet(set, graph->vertices);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// sort the edges of the graph
	sort(graph);

	if (rank == 0) {
		// add edges to the MST
		int currentEdge = 0;
		int edgesMST = 0;
		while (edgesMST < graph->vertices - 1 || currentEdge < graph->edges) {
			// check for loops if edge would be inserted
			int canonicalElementFrom = findSet(set,
					graph->edgeList[currentEdge * EDGE_MEMBERS]);
			int canonicalElementTo = findSet(set,
					graph->edgeList[currentEdge * EDGE_MEMBERS + 1]);
			if (canonicalElementFrom != canonicalElementTo) {
				// add edge to MST
				for (int i = 0; i < EDGE_MEMBERS; i++) {
					mst->edgeList[edgesMST * EDGE_MEMBERS + i] =
							graph->edgeList[currentEdge * EDGE_MEMBERS + i];
				}
				unionSet(set, canonicalElementFrom, canonicalElementTo);
				edgesMST++;
			}
			currentEdge++;
		}
	}

	// clean up
	deleteSet(set);
}

/*
 * find a MST of the graph using Prim's algorithm with a binary heap
 */
void mstPrimBinary(const WeightedGraph* graph, const WeightedGraph* mst) {
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		// create needed data structures
		AdjacencyList* list = &(AdjacencyList ) { .elements = 0, .lists = NULL };
		newAdjacencyList(list, graph);
		for (int i = 0; i < graph->edges; i++) {
			pushAdjacencyList(list, graph->edgeList[i * EDGE_MEMBERS],
					graph->edgeList[i * EDGE_MEMBERS + 1],
					graph->edgeList[i * EDGE_MEMBERS + 2]);
		}

		BinaryMinHeap* heap = &(BinaryMinHeap ) { .alloced = 0, .size = 0,
						.positions = NULL, .elements = NULL };
		newBinaryMinHeap(heap);
		heap->positions = (int*) realloc(heap->positions,
				graph->vertices * sizeof(int));
		for (int i = 0; i < graph->vertices; i++) {
			pushBinaryMinHeap(heap, i, INT_MAX, INT_MAX);
		}

		int vertex;
		int via;
		int weight;

		// start at first vertex
		decreaseBinaryMinHeap(heap, 0, 0, 0);
		popBinaryMinHeap(heap, &vertex, &via, &weight);
		for (int i = 0; i < list->lists[vertex].size; i++) {
			decreaseBinaryMinHeap(heap, list->lists[vertex].elements[i].vertex,
					vertex, list->lists[vertex].elements[i].weight);
		}

		for (int i = 0; heap->size > 0; i++) {
			// add edge from heap to MST
			popBinaryMinHeap(heap, &vertex, &via, &weight);
			mst->edgeList[i * EDGE_MEMBERS] = vertex;
			mst->edgeList[i * EDGE_MEMBERS + 1] = via;
			mst->edgeList[i * EDGE_MEMBERS + 2] = weight;

			// update heap
			for (int i = 0; i < list->lists[vertex].size; i++) {
				decreaseBinaryMinHeap(heap,
						list->lists[vertex].elements[i].vertex, vertex,
						list->lists[vertex].elements[i].weight);
			}
		}

		// clean up
		deleteBinaryMinHeap(heap);
		deleteAdjacencyList(list);
	}
}

/*
 * TODO
 * find a MST of the graph using Prim's algorithm with a fibonacci heap
 */
void mstPrimFibonacci(const WeightedGraph* graph, const WeightedGraph* mst) {
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
//		FibonacciMinHeap* heap = &(FibonacciMinHeap ) { .size = 0, .minimum =
//				NULL, .positions = NULL };
//		newFibonacciMinHeap(heap);
//		int size = 6;
//		int decrease = 4;
//		heap->positions = (FibonacciHeapElement**) realloc(heap->positions,
//				size * sizeof(FibonacciHeapElement*));
//		int vertex = -1;
//		int via = -1;
//		int weight = -1;
//
//		for (int i = 0; i < size; i++) {
//			pushFibonacciMinHeap(heap, i, i, i);
//		}
//		printFibonacciHeap(heap, heap->minimum);
//
//		popFibonacciMinHeap(heap, &vertex, &via, &weight);
//		printf("popped: %d|%d|%d\n", vertex, via, weight);
//		printFibonacciHeap(heap, heap->minimum);
//
//		heap->minimum->child->marked = 1;
//
//		decreaseFibonacciMinHeap(heap, decrease, 0, 0);
//		printf("decreased %d\n", decrease);
//		printFibonacciHeap(heap, heap->minimum);
//		for (int i = 0; i < size - 1; i++) {
//			popFibonacciMinHeap(heap, &vertex, &via, &weight);
//			printf("popped: %d|%d|%d\n", vertex, via, weight);
//			printFibonacciHeap(heap, heap->minimum);
//		}

// create needed data structures
		AdjacencyList* list = &(AdjacencyList ) { .elements = 0, .lists = NULL };
		newAdjacencyList(list, graph);
		for (int i = 0; i < graph->edges; i++) {
			pushAdjacencyList(list, graph->edgeList[i * EDGE_MEMBERS],
					graph->edgeList[i * EDGE_MEMBERS + 1],
					graph->edgeList[i * EDGE_MEMBERS + 2]);
		}

		FibonacciMinHeap* fheap = &(FibonacciMinHeap ) { .size = 0, .minimum =
				NULL, .positions = NULL };
		newFibonacciMinHeap(fheap);
		fheap->positions = (FibonacciHeapElement**) realloc(fheap->positions,
				graph->vertices * sizeof(FibonacciHeapElement*));
		for (int i = 0; i < graph->vertices; i++) {
			pushFibonacciMinHeap(fheap, i, INT_MAX, INT_MAX);
		}
		printFibonacciHeap(fheap, fheap->minimum);

		int vertex;
		int via;
		int weight;

		// start at first vertex
		decreaseFibonacciMinHeap(fheap, 0, 0, 0);
		printFibonacciHeap(fheap, fheap->minimum);
		popFibonacciMinHeap(fheap, &vertex, &via, &weight);
		for (int i = 0; i < list->lists[vertex].size; i++) {
			decreaseFibonacciMinHeap(fheap,
					list->lists[vertex].elements[i].vertex, vertex,
					list->lists[vertex].elements[i].weight);
		}

		for (int i = 0; fheap->size > 0; i++) {
			// add edge from fheap to MST
			printf("vor pop %d\n", vertex);
			printFibonacciHeap(fheap, fheap->minimum);
			popFibonacciMinHeap(fheap, &vertex, &via, &weight);
			printf("nach pop %d\n", vertex);
			printFibonacciHeap(fheap, fheap->minimum);
			mst->edgeList[i * EDGE_MEMBERS] = vertex;
			mst->edgeList[i * EDGE_MEMBERS + 1] = via;
			mst->edgeList[i * EDGE_MEMBERS + 2] = weight;

			// update fheap
			for (int i = 0; i < list->lists[vertex].size; i++) {
				decreaseFibonacciMinHeap(fheap,
						list->lists[vertex].elements[i].vertex, vertex,
						list->lists[vertex].elements[i].weight);
			}
			printf("decreased %d\n", vertex);
			printFibonacciHeap(fheap, fheap->minimum);
		}

		// clean up
		deleteFibonacciMinHeap(fheap);
		deleteAdjacencyList(list);
	}
}

/*
 * create adjacency list
 */
void newAdjacencyList(AdjacencyList* list, const WeightedGraph* graph) {
	list->elements = graph->vertices;
	list->lists = (List*) malloc(list->elements * sizeof(List));
	for (int i = 0; i < list->elements; i++) {
		list->lists[i].alloced = 1;
		list->lists[i].size = 0;
		list->lists[i].elements = (ListElement*) malloc(sizeof(ListElement));
	}
}

/*
 * create binary min heap
 */
void newBinaryMinHeap(BinaryMinHeap* heap) {
	heap->alloced = 2;
	heap->size = 0;
	heap->positions = (int*) malloc(sizeof(int));
	heap->elements = (BinaryHeapElement*) malloc(2 * sizeof(BinaryHeapElement));
}

/*
 * create fibonacci min heap element
 */
void newFibonacciHeapElement(FibonacciHeapElement* element, int vertex, int via,
		int weight, FibonacciHeapElement* left, FibonacciHeapElement* right,
		FibonacciHeapElement* parent, FibonacciHeapElement* child) {
	element->childrens = 0;
	element->marked = 0;
	element->vertex = vertex;
	element->via = via;
	element->weight = weight;
	element->parent = parent;
	element->child = child;
	element->left = left;
	element->right = right;
}

/*
 * create fibonacci min heap
 */
void newFibonacciMinHeap(FibonacciMinHeap* heap) {
	heap->size = 0;
	heap->positions = (FibonacciHeapElement**) malloc(
			sizeof(FibonacciHeapElement*));
}

/*
 * initialize and allocate memory for the members of the graph
 */
void newSet(Set* set, const int elements) {
	set->elements = elements;
	set->canonicalElements = (int*) malloc(elements * sizeof(int));
	memset(set->canonicalElements, UNSET_CANONICAL_ELEMENT,
			elements * sizeof(int));
	set->rank = (int*) malloc(elements * sizeof(int));
	memset(set->rank, 0, elements * sizeof(int));
}

/*
 * initialize and allocate memory for the members of the graph
 */
void newWeightedGraph(WeightedGraph* graph, const int vertices, const int edges) {
	graph->edges = edges;
	graph->vertices = vertices;
	graph->edgeList = (int*) malloc(edges * EDGE_MEMBERS * sizeof(int));
	memset(graph->edgeList, 0, edges * EDGE_MEMBERS * sizeof(int));
}

/*
 * remove the minimum of the heap
 */
void popBinaryMinHeap(BinaryMinHeap* heap, int* vertex, int* via, int* weight) {
	*vertex = heap->elements[0].vertex;
	*via = heap->elements[0].via;
	*weight = heap->elements[0].weight;
	heap->positions[heap->elements[0].vertex] = -1;
	heap->elements[0] = heap->elements[heap->size - 1];
	heap->positions[heap->elements[0].vertex] = 0;
	heap->size--;
	heapifyDownBinaryMinHeap(heap, 0);
}

/*
 * remove the minimum of the heap
 */
void popFibonacciMinHeap(FibonacciMinHeap* heap, int* vertex, int* via,
		int* weight) {
	if (heap->minimum != NULL) {
		FibonacciHeapElement* child = heap->minimum->child;
		*vertex = heap->minimum->vertex;
		*via = heap->minimum->via;
		*weight = heap->minimum->weight;

		if (heap->minimum->child != NULL) {
			// make children of minimum to root elements
			FibonacciHeapElement* startElement = child;
			FibonacciHeapElement* element = child;

			// remove parent pointer
			do {
				element->parent = NULL;
				element = element->right;
			} while (element != startElement);

			// add children to root list left to the minimum
			FibonacciHeapElement* endHeap = heap->minimum->left;
			FibonacciHeapElement* endChildren = child->left;
			endChildren->right = heap->minimum;
			heap->minimum->left = endChildren;
			endHeap->right = child;
			child->left = endHeap;
		}
		// update right/left
		heap->minimum->right->left = heap->minimum->left;
		heap->minimum->left->right = heap->minimum->right;

		// remove minimum
		heap->positions[heap->minimum->vertex] = NULL;
		heap->size--;
		FibonacciHeapElement* toFree = heap->minimum;
		heap->minimum = heap->minimum->right;
		free(toFree);

		printf("in pop before consolidate %d\n", *vertex);
		printFibonacciHeap(heap, heap->minimum);

		if (heap->size > 0) {
			// rearrange heap and update minimum
			consolidateHeap(heap);
		}
	}
}

/*
 * prints the adjacency list
 */
void printAdjacencyList(const AdjacencyList* list) {
	for (int i = 0; i < list->elements; i++) {
		printf("%d:", i);
		for (int j = 0; j < list->lists[i].size; j++) {
			printf(" %d(%d)", list->lists[i].elements[j].vertex,
					list->lists[i].elements[j].weight);
		}
		printf("\n");
	}
}

/*
 * print binary min heap
 */
void printBinaryHeap(const BinaryMinHeap* heap) {
	for (int i = 0; i < heap->size; i++) {
		printf("[P%d]%d: %d(%d) ", heap->positions[heap->elements[i].vertex],
				heap->elements[i].vertex, heap->elements[i].via,
				heap->elements[i].weight);
		if (log2(i + 2) == (int) log2(i + 2)) {
			// line break after each stage
			printf("\n");
		}
	}
	printf("\n");
}

/*
 * print fibonacci min heap
 */
void printFibonacciHeap(const FibonacciMinHeap* heap,
		FibonacciHeapElement* startElement) {
	if (heap->size > 0) {
		FibonacciHeapElement* currentElement = startElement;
		printf("[%d]:", startElement->vertex);
		do {
			printf(" (%d,%d)%d|%d|%d", currentElement->marked,
					currentElement->childrens, currentElement->vertex,
					currentElement->via, currentElement->weight);
			currentElement = currentElement->right;
		} while (currentElement != startElement);
		printf("\n");
		do {
			if (currentElement->child != NULL) {
				printf("{%d}", currentElement->vertex);
				printFibonacciHeap(heap, currentElement->child);
				printf("\n");
			}
			currentElement = currentElement->right;
		} while (currentElement != startElement);
	} else {
		printf("heap is empty!\n");
	}
}

/*
 * print the graph as a maze to console
 */
void printMaze(const WeightedGraph* graph, int rows, int columns) {
	// initialize the maze with spaces
	int rowsMaze = rows * 2 - 1;
	int columnsMaze = columns * 2 - 1;
	char maze[rowsMaze][columnsMaze];
	memset(maze, ' ', rowsMaze * columnsMaze * sizeof(char));

	// each vertex is represented as a plus sign
	for (int i = 0; i < rowsMaze; i++) {
		for (int j = 0; j < columnsMaze; j++) {
			if (i % 2 == 0 && j % 2 == 0) {
				maze[i][j] = '+';
			}
		}
	}

	// each edge is represented as dash or pipe sign
	for (int i = 0; i < graph->edges; i++) {
		int from;
		int to;
		if (graph->edgeList[i * EDGE_MEMBERS]
				< graph->edgeList[i * EDGE_MEMBERS + 1]) {
			from = graph->edgeList[i * EDGE_MEMBERS];
			to = graph->edgeList[i * EDGE_MEMBERS + 1];
		} else {
			to = graph->edgeList[i * EDGE_MEMBERS];
			from = graph->edgeList[i * EDGE_MEMBERS + 1];
		}
		int row = from / columns + to / columns;
		if ((row % 2)) {
			// edges in even rows are displayed as pipes
			maze[row][(to % columns) * 2] = '|';
		} else {
			// edges in uneven rows are displayed as dashes
			maze[row][(to % columns - 1) * 2 + 1] = '-';
		}
	}

	// print the char array to the console
	for (int i = 0; i < rowsMaze; i++) {
		for (int j = 0; j < columnsMaze; j++) {
			printf("%c", maze[i][j]);
		}
		printf("\n");
	}
}

/*
 * print the components of the set
 */
void printSet(const Set* set) {
	for (int i = 0; i < set->elements; i++) {
		printf("%d: %d(%d)\n", i, set->canonicalElements[i], set->rank[i]);
	}
}

/*
 * print all edges of the graph in "from to weight" format
 */
void printWeightedGraph(const WeightedGraph* graph) {
	for (int i = 0; i < graph->edges; i++) {
		for (int j = 0; j < EDGE_MEMBERS; j++) {
			printf("%d\t", graph->edgeList[i * EDGE_MEMBERS + j]);
		}
		printf("\n");
	}
}

/*
 * process the command line parameters and return a Handle struct with them
 */
Handle processParameters(int argc, char* argv[]) {
	Handle handle = { .algorithm = 0, .columns = 3, .maze = 0, .new = 0,
			.parallel = 0, .rows = 2, .verbose = 0 };

	if (argc > 1) {
		while ((argc > 1) && (argv[1][0] == '-')) {
			switch (argv[1][1]) {

			case 'a':
				// choose algorithm
				handle.algorithm = atoi(&argv[2][0]);
				++argv;
				--argc;
				break;

			case 'c':
				// set number of columns
				handle.columns = atoi(&argv[2][0]);
				++argv;
				--argc;
				break;

			case 'h':
				// print help message
				printf(
						"Parameters:\n"
								"\t-a <int>\tchoose algorithm: 0 Kruskal (default), 1 Prim (Fibonacci), 2 Prim (Binary), 3 Boruvka\n"
								"\t-c <int>\tset number of columns (default: 3)\n"
								"\t-h\t\tprint this help message\n"
								"\t-m\t\tprint the resulting maze to console at the end\n"
								"\t-n\t\tcreate a new maze file\n"
								"\t-r <int>\tset number of rows (default: 2)\n"
								"\t-v\t\tprint more information\n"
								"\nThis program is distributed under the terms of the LGPLv3 license\n");
				handle.help = 1;
				break;

			case 'm':
				// print the resulting maze to console at the end
				handle.maze = 1;
				break;

			case 'n':
				// create a new maze file
				handle.new = 1;
				break;

			case 'p':
				// run in parallel
				handle.parallel = 1;
				break;

			case 'r':
				// set number of rows
				handle.rows = atoi(&argv[2][0]);
				++argv;
				--argc;
				break;

			case 'v':
				// print more information
				handle.verbose = 1;
				break;

			default:
				printf("Wrong parameter: %s\n", argv[1]);
				printf("-h for help\n");
				exit(1);
			}

			++argv;
			--argc;
		}
	}

	return handle;
}

/*
 * add edge to adjacency list
 */
void pushAdjacencyList(AdjacencyList* list, int from, int to, int weight) {
	// double the size if adjacency list is full
	if (list->lists[from].size == list->lists[from].alloced) {
		list->lists[from].elements = (ListElement*) realloc(
				list->lists[from].elements,
				2 * list->lists[from].alloced * sizeof(ListElement));
		list->lists[from].alloced *= 2;
	}

	// add element at the end
	list->lists[from].elements[list->lists[from].size] = (ListElement ) {
					.vertex = to, .weight = weight };
	list->lists[from].size++;

	// same for the other vertex
	if (list->lists[to].size == list->lists[to].alloced) {
		list->lists[to].elements = (ListElement*) realloc(
				list->lists[to].elements,
				2 * list->lists[to].alloced * sizeof(ListElement));
		list->lists[to].alloced *= 2;
	}

	list->lists[to].elements[list->lists[to].size] = (ListElement ) { .vertex =
					from, .weight = weight };
	list->lists[to].size++;
}

/*
 * push a new element to the end of a binary heap, then bubble up
 */
void pushBinaryMinHeap(BinaryMinHeap* heap, const int vertex, const int via,
		const int weight) {
	if (heap->size == heap->alloced) {
		// double the size if heap is full
		heap->elements = (BinaryHeapElement*) realloc(heap->elements,
				2 * heap->alloced * sizeof(BinaryHeapElement));
		heap->alloced *= 2;
	}

	heap->elements[heap->size] = (BinaryHeapElement ) { .vertex = vertex, .via =
					via, .weight = weight };
	heap->positions[vertex] = heap->size;

	heapifyBinaryMinHeap(heap, heap->size);

	heap->size++;
}

/*
 * add a new element
 */
void pushFibonacciMinHeap(FibonacciMinHeap* heap, const int vertex,
		const int via, const int weight) {
	FibonacciHeapElement* element = (FibonacciHeapElement*) malloc(
			sizeof(FibonacciHeapElement));
	newFibonacciHeapElement(element, vertex, via, weight, element, element,
	NULL, NULL);
	heap->positions[element->vertex] = element;

	// insert as root element
	insertFibonacciMinHeap(heap, element);
	heap->size++;
}

/*
 * read a previously generated maze file and store it in the graph
 */
void readMazeFile(WeightedGraph* graph, const char inputFileName[]) {
	// open the file
	FILE* inputFile;
	const char* inputMode = "r";
	inputFile = fopen(inputFileName, inputMode);
	if (inputFile == NULL) {
		printf("Could not open input file, exiting!\n");
		exit(1);
	}

	int fscanfResult;

	// first line contains number of vertices and edges
	int vertices = 0;
	int edges = 0;
	fscanfResult = fscanf(inputFile, "%d %d", &vertices, &edges);
	newWeightedGraph(graph, vertices, edges);

	// all lines after the first contain the edges
	// values stored as "from to weight"
	int from;
	int to;
	int weight;
	for (int i = 0; i < edges; i++) {
		fscanfResult = fscanf(inputFile, "%d %d %d", &from, &to, &weight);
		graph->edgeList[i * EDGE_MEMBERS] = from;
		graph->edgeList[i * EDGE_MEMBERS + 1] = to;
		graph->edgeList[i * EDGE_MEMBERS + 2] = weight;
	}

	fclose(inputFile);

	// EOF result of *scanf indicates an error
	if (fscanfResult == EOF) {
		printf("Something went wrong during reading of maze file, exiting!\n");
		exit(1);
	}
}

/*
 * sort the edges of the graph in parallel with mergesort in parallel
 */
void sort(WeightedGraph* graph) {
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status status;

	// send number of elements
	int elements;
	if (rank == 0) {
		elements = graph->edges;
		MPI_Bcast(&elements, 1, MPI_INT, 0, MPI_COMM_WORLD);
	} else {
		MPI_Bcast(&elements, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}

	// scatter the edges to sort
	int elementsPart = (elements + size - 1) / size;
	int* edgeListPart = (int*) malloc(
			elementsPart * EDGE_MEMBERS * sizeof(int));
	MPI_Scatter(graph->edgeList, elementsPart * EDGE_MEMBERS, MPI_INT,
			edgeListPart, elementsPart * EDGE_MEMBERS,
			MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == size - 1 && elements % elementsPart != 0) {
		// number of elements and processes isn't divisible without remainder
		elementsPart = elements % elementsPart;
	}

	if (elements / 2 + 1 < size && elements != size) {
		if (rank == 0) {
			printf("Unsupported size/process combination, exiting!\n");
		}
		MPI_Finalize();
		exit(1);
	}

	// sort the part
	mergeSort(edgeListPart, 0, elementsPart - 1);

	// merge all parts
	int from;
	int to;
	int elementsRecieved;
	for (int step = 1; step < size; step *= 2) {
		if (rank % (2 * step) == 0) {
			from = rank + step;
			if (from < size) {
				MPI_Recv(&elementsRecieved, 1, MPI_INT, from, 0,
				MPI_COMM_WORLD, &status);
				edgeListPart = realloc(edgeListPart,
						(elementsPart + elementsRecieved) * EDGE_MEMBERS
								* sizeof(int));
				MPI_Recv(&edgeListPart[elementsPart * EDGE_MEMBERS],
						elementsRecieved * EDGE_MEMBERS,
						MPI_INT, from, 0, MPI_COMM_WORLD, &status);
				merge(edgeListPart, 0, elementsPart + elementsRecieved - 1,
						elementsPart - 1);
				elementsPart += elementsRecieved;
			}
		} else if (rank % (step) == 0) {
			to = rank - step;
			MPI_Send(&elementsPart, 1, MPI_INT, to, 0, MPI_COMM_WORLD);
			MPI_Send(edgeListPart, elementsPart * EDGE_MEMBERS, MPI_INT, to, 0,
			MPI_COMM_WORLD);
		}
	}

	// edgeListPart is the new edgeList of the graph, cleanup other memory
	if (rank == 0) {
		free(graph->edgeList);
		graph->edgeList = edgeListPart;
	} else {
		free(edgeListPart);
	}
}

/*
 * helper function to swap binary heap elements
 */
void swapBinaryHeapElement(BinaryMinHeap* heap, const int position1,
		const int position2) {
	heap->positions[heap->elements[position1].vertex] = position2;
	heap->positions[heap->elements[position2].vertex] = position1;

	BinaryHeapElement swap = heap->elements[position1];
	heap->elements[position1] = heap->elements[position2];
	heap->elements[position2] = swap;
}

/*
 * TODO check
 * update the degree pointers
 */
FibonacciHeapElement* updateDegreeFibonacciMinHeap(
		FibonacciHeapElement** degree, FibonacciHeapElement* parent,
		FibonacciHeapElement* child) {
	child->right->left = child->left;
	child->left->right = child->right;
	if (parent->child == NULL) {
		// parent had no child
		child->right = child;
		child->left = child;
	} else {
		// parent had a child
		child->right = parent->child;
		child->left = parent->child->left;
		parent->child->left->right = child;
		parent->child->left = child;
	}
	child->parent = parent;
	parent->child = child;

	// update degree
	parent->childrens++;
	child->marked = 0;
	degree[parent->childrens - 1] = NULL;
	if (degree[parent->childrens] == NULL) {
		// no tree with same degree
		degree[parent->childrens] = parent;
		return parent;
	} else {
		// tree with same degree, combine
		if (degree[parent->childrens]->weight > parent->weight) {
			return updateDegreeFibonacciMinHeap(degree, parent,
					degree[parent->childrens]);
		} else {
			return updateDegreeFibonacciMinHeap(degree,
					degree[parent->childrens], parent);
		}
	}
}

/*
 * merge the set of parent1 and parent2 with union by rank
 */
void unionSet(Set* set, const int parent1, const int parent2) {
	int root1 = findSet(set, parent1);
	int root2 = findSet(set, parent2);

	if (root1 == root2) {
		return;
	} else if (set->rank[root1] < set->rank[root2]) {
		set->canonicalElements[root1] = root2;
	} else if (set->rank[root1] > set->rank[root2]) {
		set->canonicalElements[root2] = root1;
	} else {
		set->canonicalElements[root1] = root2;
		set->rank[root2] = set->rank[root1] + 1;
	}
}

minimum-spanning-tree
=====================

General
-------

This project aims to implement the algorithms of Kruskal, Prim and Boruvka for creating a minimum spanning tree (MST) of a weighted, undirected graph in C with parallelization via MPI.

Applications of MSTs
--------------------

A spanning tree can be used for various application:
* Spanning Tree Protocol (STP) in bridged Ethernet networks to ensure loop free connections
* minimum cost networks (roads, electricity, telephone, ...)
* approximation of the travelling salesman problem (TSP)
* generation of mazes

Status
------

At the moment Kruskal's and Prim's algorithms are implemented.

Output
------

To test the program the maze generation was choosen. The program generates a 2D grid graph with random edge weights. The resulting maze can be printed to the console by passing the argument `-m` to the program. A `+` represents a vertex and `-`,`|` represent an edge between two vertices.

Example output (`mpirun -np 1 ./MST -c 12 -r 8 -a 0 -n -m`):
```
Starting
Time for sorting: 0.000020 s
Time elapsed: 0.000037 s
MST weight: 481
Maze:
+-+-+-+ +-+ +-+-+ +-+ +
| |     | |     | |   |
+ + + + + +-+-+ + + +-+
  | | |     |   | |   |
+-+-+-+-+-+-+-+-+-+ + +
| |     |   |       | |
+ +-+ +-+ + + + + + + +
|   | |   | | | | | | |
+ +-+ + +-+ +-+-+ + +-+
|   |     |   |   |   |
+-+ +-+ +-+-+-+ + +-+ +
|         | |   |   | |
+ +-+-+-+-+ + + +-+-+-+
    | | | |   | |   | |
+-+-+ + + +-+-+-+ +-+ +
Finished
```

Parameters
----------

```
-a <int>  choose algorithm: 0 Kruskal (default), 1 Prim, 2 Boruvka
-c <int>  set number of columns (default: 3)
-h        print this help message
-m        print the resulting maze to console at the end
-n        create a new maze file
-r <int>  set number of rows (default: 2)
-v        print more information
```

Implementation overview
-----------------------

Kruskal's algorithm:
* sorting edges via parallelized mergesort
* track components via union-find data structure with union by rank and path compression

Prim's algorithm:
* store neighbors in adjacency list
* track vertices in binary heap

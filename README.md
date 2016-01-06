minimum-spanning-tree
=====================

This project implements the algorithms of Kruskal, Prim and Boruvka for creating a minimum spanning tree (MST) of a weighted, undirected graph in C with parallelization via MPI. It was developed for the module "Algorithm Engineering" at the HTWK Leipzig. The work was also submitted and accepted as a paper for [SKILL 2015](http://skill.informatik.uni-leipzig.de/blog/historie/skill2015/) which was part of [INFORMATIK 2015](http://www.informatik2015.de/). A [copy](paper/paper-german.pdf) and the corresponding [BibTeX file](paper/Jahne15.bib) can be found in the directory `paper`. I will add a translation into English somewhere in the future.

Applications of MSTs
--------------------

A spanning tree can be used for various applications:
- Spanning Tree Protocol (STP) in bridged Ethernet networks to ensure loop free connections
- minimum cost networks (roads, electricity, telephone, ...)
- approximation of the traveling salesman problem (TSP)
- generation of mazes
- ...

Status
------

The algorithms of Kruskal, Prim and Boruvka are implemented. Kruskal's and Boruvka's algorithm are partially parallelized. Kruskal's has a parallel merge sort for sorting the the edge list and Boruvka's searches in parallel for the minimum outgoing edges of each component.

Output
------

The maze generation was chosen, to test the program. The program can generate a 2D square grid graph with random edge weights. The resulting maze can be printed to the console by passing the argument `-m` to the program. A `+` represents a vertex and `-`, `|` represent an edge between two vertices.

Example output (`mpirun -np 1 ./MST -c 12 -r 8 -a 0 -n -f mazeGraph.csv -m`):
```
Starting
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
-a <int>  choose algorithm: 0 Kruskal (default), 1 Prim (fibonacci), 2 Prim (binary), 3 Boruvka
-c <int>  set number of columns (default: 3)
-f <path> file to store and read graph from (default: maze.csv)
-h        print this help message
-m        print the resulting maze to console at the end (correct number of rows and columns needed!)
-n        create a new maze file
-r <int>  set number of rows (default: 2)
-v        print more information
```

Implementation overview
-----------------------

Kruskal's algorithm:
- sorting edges via parallelized merge sort
- track components via union-find data structure with union by rank and path compression

Prim's algorithm:
- store neighbors in adjacency list
- track vertices in binary or fibonacci heap

Boruvka's algorithm:
- track components via union-find data structure with union by rank and path compression
- parallelized search for minimum outgoing edge

Results
-------

For benchmarking purposes the graphs of the [9. DIMACS Implementation Challenge](http://www.dis.uniroma1.it/challenge9/download.shtml) were chosen. Generic graphs were generated to make statements on dense graphs. These had 10000 vertices and varying number of edges with random edge weights below 100.

It could be shown that Boruvka's algorithm runs faster then Kruskal's and Prim's with both heaps. Additionally it uses the least amount of memory and is comparatively easy to implement. It also outperformed the other algorithms on dense graphs and had the least increase of runtime. The chosen parallelization was only suitable for medium dense graphs. It actually slowed down the runtime for sparse graphs.

Kruskal's algorithm could be shown to be the second choice for sparse graphs. But it has the worst running time on dense graph. It was able to outperform Boruvka's algorithm on sparse graphs with the parallelized merge sort and enough processors. The memory usage is medium.

Prim's algorithm with a fibonacci heap was only able to outperform the binary heap slightly for very large graphs. In all other cases the implementation with the binary heap was faster. It also uses the most amount of memory and was the most complicated to implement. Therefore is the use of a fibonacci heap discouraged. Prim's algorithm with the binary heap was on pair with Kruskal's algorithm for small graphs and uses more memory.

In conclusion the use of Boruvka's algorithm is recommended to find the MST of a graph.

Maze visualization
------------------

The MST of a graph can be interpreted as a maze. With the Java program created from [src/Maze2PNG.java](src/Maze2PNG.java) you can visualize a square grid graph created with he `-m` option. Just write the output in a file: `mpirun -np 1 ./MST -c 12 -r 8 -a 0 -n -f mazeGraph.csv -m > maze.txt` and pass it to the program as the first argument. The second argument specifies the name of the PNG file to safe the visualization: `java Maze2PNG maze.txt maze.png`. Each `+`, `-` and `|` are represented as white pixels, spaces are black pixels. There is also a one pixel wide black padding around the maze. The image generated out of the example above can be seen in the following screenshot.

![maze.png](maze.png)
all: run

build: boruvka.c
	gcc -o boruvka boruvka.c

run: build
	./boruvka

display: run
	sfdp -Tpng graph.dot -o graph0.png
	fdp -Tpng graph.dot -o graph1.png
	dot -Tpng graph.dot -o graph2.png
	neato -Tpng graph.dot -o graph3.png
	twopi -Tpng graph.dot -o graph4.png
	circo -Tpng graph.dot -o graph5.png
	rm graph.dot

clean: 
	rm boruvka graph.dot mst.txt graph.txt *.png
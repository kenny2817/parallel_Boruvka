all: run

build: boruvka.c
	gcc -o boruvka boruvka.c

run: build
	./boruvka

display: run
	sfdp -Tpng graph.dot -o graph.png
	rm graph.dot

clean: 
	rm boruvka graph.dot mst.txt graph.png
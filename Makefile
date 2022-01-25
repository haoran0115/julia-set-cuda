CC=nvcc
julia: julia.cu
	$(CC) julia.cu -o julia
run:
	make
	./julia
clean:
	rm julia


# -std=c++14: we're limiting ourselves to c++14, since that's what the 
#             GCC compiler on the VSC supports.
# -DNDEBUG: turns off e.g. assertion checks
# -O3: enables optimizations in the compiler

# Settings for optimized build
FLAGS=-O3 -DNDEBUG -std=c++14

# Settings for a debug. build
# FLAGS=-g -std=c++14

all: kmeans

clean:
	rm -f kmeans

kmeans: main_startcode.cpp rng.cpp
	$(CXX) $(FLAGS) -o my_serial_kmeans main_startcode.cpp rng.cpp
	./my_serial_kmeans --input mouse_500x2.csv --output output.csv --k 3 --repetitions 10 --seed 1338 --centroidtrace centroidtrace.csv --trace clustertrace.csv

compare: 
	./compare.py ./my_serial_kmeans /data/leuven/303/vsc30380/kmeans_serial_reference

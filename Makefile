# -std=c++14: we're limiting ourselves to c++14, since that's what the 
#             GCC compiler on the VSC supports.
# -DNDEBUG: turns off e.g. assertion checks
# -O3: enables optimizations in the compiler

# Settings for optimized build
FLAGS=-O3 -DNDEBUG -std=c++14

# Settings for a debug build
# FLAGS=-g -std=c++14

OUTPUT_SERIAL_PATH = outputSerial
OUTPUT_OMP_PATH = outputOmp
COMPARE_PATH = compare
DATA_PATH = data
DATA1 = mouse_500x2
DATA2 = Frogs_MFCCs_7195x22
DATA3 = 100000x5

SERIAL_EX = my_serial_kmeans
OMP_EX = my_omp_kmeans

THREADS = 1

all: serial

clean:
	rm -f $(SERIAL_EX) $(OMP_EX) outputOmp/* outputSerial/*

serial: clean mainSerial.cpp rng.cpp
	$(CXX) $(FLAGS) -o $(SERIAL_EX) mainSerial.cpp rng.cpp
	@echo
	@echo "SERIAL VERSION:"
	@echo
	@echo "---------------------------------------------------------------------------------------------"
	@echo
	./$(SERIAL_EX) --input $(DATA_PATH)/$(DATA1).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA1).csv --seed 1337 --k 10 --repetitions 20 --centroidtrace $(OUTPUT_SERIAL_PATH)/centroidT$(DATA1).csv --trace $(OUTPUT_SERIAL_PATH)/clusterT$(DATA1).csv
	@echo
	/data/leuven/303/vsc30380/kmeans_serial_reference --input $(DATA_PATH)/$(DATA1).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA1)Ref.csv --seed 1337 --k 10 --repetitions 20 --centroidtrace $(OUTPUT_SERIAL_PATH)/centroidT$(DATA1)Ref.csv --trace $(OUTPUT_SERIAL_PATH)/clusterT$(DATA1)Ref.csv
	@echo
	@echo "---------------------------------------------------------------------------------------------"
	@echo
	./$(SERIAL_EX) --input $(DATA_PATH)/$(DATA2).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA2).csv --seed 1337 --k 10 --repetitions 20 --centroidtrace $(OUTPUT_SERIAL_PATH)/centroidT$(DATA2).csv --trace $(OUTPUT_SERIAL_PATH)/clusterT$(DATA2).csv
	@echo
	/data/leuven/303/vsc30380/kmeans_serial_reference --input $(DATA_PATH)/$(DATA2).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA2)Ref.csv --seed 1337 --k 10 --repetitions 20 --centroidtrace $(OUTPUT_SERIAL_PATH)/centroidT$(DATA2)Ref.csv --trace $(OUTPUT_SERIAL_PATH)/clusterT$(DATA2)Ref.csv
	@echo
	@echo "---------------------------------------------------------------------------------------------"
	@echo
	./$(SERIAL_EX) --input $(DATA_PATH)/$(DATA3).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA3).csv --seed 1337 --k 10 --repetitions 20 --centroidtrace $(OUTPUT_SERIAL_PATH)/centroidT$(DATA3).csv --trace $(OUTPUT_SERIAL_PATH)/clusterT$(DATA3).csv
	@echo
	/data/leuven/303/vsc30380/kmeans_serial_reference --input $(DATA_PATH)/$(DATA3).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA3)Ref.csv --seed 1337 --k 10 --repetitions 20 --centroidtrace $(OUTPUT_SERIAL_PATH)/centroidT$(DATA3)Ref.csv --trace $(OUTPUT_SERIAL_PATH)/clusterT$(DATA3)Ref.csv

omp: clean mainOmp.cpp rng.cpp
	$(CXX) $(FLAGS) -o $(OMP_EX) -fopenmp  mainOmp.cpp rng.cpp
	
	@echo
	@echo "OMP VERSION:"
	@echo
	@echo "---------------------------------------------------------------------------------------------"
	@echo
	./$(OMP_EX) --threads $(THREADS) --input $(DATA_PATH)/$(DATA1).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA1).csv --seed 1337 --k 10 --repetitions 20 --centroidtrace $(OUTPUT_SERIAL_PATH)/centroidT$(DATA1).csv --trace $(OUTPUT_SERIAL_PATH)/clusterT$(DATA1).csv
	@echo
	/data/leuven/303/vsc30380/kmeans_serial_reference --input $(DATA_PATH)/$(DATA1).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA1)Ref.csv --seed 1337 --k 10 --repetitions 20 --centroidtrace $(OUTPUT_SERIAL_PATH)/centroidT$(DATA1)Ref.csv --trace $(OUTPUT_SERIAL_PATH)/clusterT$(DATA1)Ref.csv
	@echo
	@echo "---------------------------------------------------------------------------------------------"
	@echo
	./$(OMP_EX) --threads $(THREADS) --input $(DATA_PATH)/$(DATA2).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA2).csv --seed 1337 --k 10 --repetitions 20 --centroidtrace $(OUTPUT_SERIAL_PATH)/centroidT$(DATA2).csv --trace $(OUTPUT_SERIAL_PATH)/clusterT$(DATA2).csv
	@echo
	/data/leuven/303/vsc30380/kmeans_serial_reference --input $(DATA_PATH)/$(DATA2).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA2)Ref.csv --seed 1337 --k 10 --repetitions 20 --centroidtrace $(OUTPUT_SERIAL_PATH)/centroidT$(DATA2)Ref.csv --trace $(OUTPUT_SERIAL_PATH)/clusterT$(DATA2)Ref.csv
	@echo
	@echo "---------------------------------------------------------------------------------------------"
	@echo
	./$(OMP_EX) --threads $(THREADS) --input $(DATA_PATH)/$(DATA3).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA3).csv --seed 1337 --k 10 --repetitions 20 --centroidtrace $(OUTPUT_SERIAL_PATH)/centroidT$(DATA3).csv --trace $(OUTPUT_SERIAL_PATH)/clusterT$(DATA3).csv
	@echo
	/data/leuven/303/vsc30380/kmeans_serial_reference --input $(DATA_PATH)/$(DATA3).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA3)Ref.csv --seed 1337 --k 10 --repetitions 20 --centroidtrace $(OUTPUT_SERIAL_PATH)/centroidT$(DATA3)Ref.csv --trace $(OUTPUT_SERIAL_PATH)/clusterT$(DATA3)Ref.csv

versus: serial omp
    
compare:
	./compare.py ./my_serial_kmeans /data/leuven/303/vsc30380/kmeans_serial_reference --input $(DATA_PATH)/$(DATA2).csv --output $(COMPARE_PATH)/compare.csv --seed 1337 --k 10 --repetitions 20 --centroidtrace $(COMPARE_PATH)/centroidTrace.csv --trace $(COMPARE_PATH)/clusterTrace.csv

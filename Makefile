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
DATA4 = drivFaceD_606x6400
DATA5 = 1M_1000000x4

SERIAL_EX = my_serial_kmeans
OMP_EX = my_omp_kmeans

THREADS = $(filter-out $@,$(MAKECMDGOALS))



all: serial

cleanSerial:
	rm -f $(SERIAL_EX) outputSerial/*

cleanOmp:
	rm -f $(OMP_EX) outputOmp/*

test: mainOmp.cpp rng.cpp

	$(CXX) $(FLAGS) -o $(OMP_EX) -fopenmp mainOmp.cpp rng.cpp
	$(eval DATA=$(DATA3))
	#./$(SERIAL_EX) --input $(DATA_PATH)/$(DATA4).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA4).csv --seed 1337 --k 10 --repetitions 20
	./$(OMP_EX) --threads $(THREADS) --input $(DATA_PATH)/$(DATA).csv --output $(OUTPUT_OMP_PATH)/output$(DATA).csv --seed 1337 --k 10 --repetitions 20
	#/data/leuven/303/vsc30380/kmeans_serial_reference --input $(DATA_PATH)/$(DATA3).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA3)Ref.csv --seed 1337 --k 10 --repetitions 20
	#/data/leuven/303/vsc30380/kmeans_serial_reference --threads $(THREADS) --input $(DATA_PATH)/$(DATA).csv --output $(OUTPUT_OMP_PATH)/output$(DATA)Ref.csv --seed 1337 --k 10 --repetitions 20

serialAll: cleanSerial mainSerial.cpp rng.cpp
	$(eval THREADS=)
	$(CXX) $(FLAGS) -o $(SERIAL_EX) mainSerial.cpp rng.cpp
	@echo
	@echo "SERIAL VERSION:"
	@echo
	@echo "---------------------------------------------------------------------------------------------"
	@echo
	./$(SERIAL_EX) --input $(DATA_PATH)/$(DATA1).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA1).csv --seed 1337 --k 10 --repetitions 20
	@echo
	/data/leuven/303/vsc30380/kmeans_serial_reference --input $(DATA_PATH)/$(DATA1).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA1)Ref.csv --seed 1337 --k 10 --repetitions 20
	@echo
	@echo "---------------------------------------------------------------------------------------------"
	@echo
	./$(SERIAL_EX) --input $(DATA_PATH)/$(DATA2).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA2).csv --seed 1337 --k 10 --repetitions 20
	@echo
	/data/leuven/303/vsc30380/kmeans_serial_reference --input $(DATA_PATH)/$(DATA2).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA2)Ref.csv --seed 1337 --k 10 --repetitions 20
	@echo
	@echo "---------------------------------------------------------------------------------------------"
	@echo
	./$(SERIAL_EX) --input $(DATA_PATH)/$(DATA3).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA3).csv --seed 1337 --k 10 --repetitions 20
	@echo
	/data/leuven/303/vsc30380/kmeans_serial_reference --input $(DATA_PATH)/$(DATA3).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA3)Ref.csv --seed 1337 --k 10 --repetitions 20
	@echo
	@echo "---------------------------------------------------------------------------------------------"
	@echo
	./$(SERIAL_EX) --input $(DATA_PATH)/$(DATA4).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA4).csv --seed 1337 --k 10 --repetitions 20
	@echo
	/data/leuven/303/vsc30380/kmeans_serial_reference --input $(DATA_PATH)/$(DATA4).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA4)Ref.csv --seed 1337 --k 10 --repetitions 20
	@echo
	@echo "---------------------------------------------------------------------------------------------"
	@echo
	./$(SERIAL_EX) --input $(DATA_PATH)/$(DATA5).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA5).csv --seed 1337 --k 10 --repetitions 20
	@echo
	/data/leuven/303/vsc30380/kmeans_serial_reference --input $(DATA_PATH)/$(DATA5).csv --output $(OUTPUT_SERIAL_PATH)/output$(DATA5)Ref.csv --seed 1337 --k 10 --repetitions 20

ompAll: cleanOmp mainOmp.cpp rng.cpp
	$(CXX) $(FLAGS) -o $(OMP_EX) -fopenmp  mainOmp.cpp rng.cpp
	
	@echo
	@echo "OMP VERSION:"
	@echo
	@echo "---------------------------------------------------------------------------------------------"
	@echo
	./$(OMP_EX) --threads $(THREADS) --input $(DATA_PATH)/$(DATA1).csv --output $(OUTPUT_OMP_PATH)/output$(DATA1).csv --seed 1337 --k 10 --repetitions 20 --centroidtrace $(OUTPUT_OMP_PATH)/centroidT$(DATA1).csv --trace $(OUTPUT_OMP_PATH)/clusterT$(DATA1).csv
	@echo
	/data/leuven/303/vsc30380/kmeans_serial_reference --threads $(THREADS) --input $(DATA_PATH)/$(DATA1).csv --output $(OUTPUT_OMP_PATH)/output$(DATA1)Ref.csv --seed 1337 --k 10 --repetitions 20 --centroidtrace $(OUTPUT_OMP_PATH)/centroidT$(DATA1)Ref.csv --trace $(OUTPUT_OMP_PATH)/clusterT$(DATA1)Ref.csv
	@echo
	@echo "---------------------------------------------------------------------------------------------"
	@echo
	./$(OMP_EX) --threads $(THREADS) --input $(DATA_PATH)/$(DATA2).csv --output $(OUTPUT_OMP_PATH)/output$(DATA2).csv --seed 1337 --k 10 --repetitions 20 --centroidtrace $(OUTPUT_OMP_PATH)/centroidT$(DATA2).csv --trace $(OUTPUT_OMP_PATH)/clusterT$(DATA2).csv
	@echo
	/data/leuven/303/vsc30380/kmeans_serial_reference --threads $(THREADS) --input $(DATA_PATH)/$(DATA2).csv --output $(OUTPUT_OMP_PATH)/output$(DATA2)Ref.csv --seed 1337 --k 10 --repetitions 20 --centroidtrace $(OUTPUT_OMP_PATH)/centroidT$(DATA2)Ref.csv --trace $(OUTPUT_OMP_PATH)/clusterT$(DATA2)Ref.csv
	@echo
	@echo "---------------------------------------------------------------------------------------------"
	@echo
	./$(OMP_EX) --threads $(THREADS) --input $(DATA_PATH)/$(DATA3).csv --output $(OUTPUT_OMP_PATH)/output$(DATA3).csv --seed 1337 --k 10 --repetitions 20 --centroidtrace $(OUTPUT_OMP_PATH)/centroidT$(DATA3).csv --trace $(OUTPUT_OMP_PATH)/clusterT$(DATA3).csv
	@echo
	/data/leuven/303/vsc30380/kmeans_serial_reference --threads $(THREADS) --input $(DATA_PATH)/$(DATA3).csv --output $(OUTPUT_OMP_PATH)/output$(DATA3)Ref.csv --seed 1337 --k 10 --repetitions 20 --centroidtrace $(OUTPUT_OMP_PATH)/centroidT$(DATA3)Ref.csv --trace $(OUTPUT_OMP_PATH)/clusterT$(DATA3)Ref.csv

both: serialAll ompAll
    
compare:
	./compare.py ./my_serial_kmeans /data/leuven/303/vsc30380/kmeans_serial_reference --input $(DATA_PATH)/$(DATA2).csv --output $(COMPARE_PATH)/compare.csv --seed 1337 --k 10 --repetitions 20 --centroidtrace $(COMPARE_PATH)/centroidTrace.csv --trace $(COMPARE_PATH)/clusterTrace.csv

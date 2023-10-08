#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include "CSVReader.hpp"
#include "CSVWriter.hpp"
#include "rng.h"
#include "timer.h"

void usage()
{
	std::cerr << R"XYZ(
Usage:

  kmeans --input inputfile.csv --output outputfile.csv --k numclusters --repetitions numrepetitions --seed seed [--blocks numblocks] [--threads numthreads] [--trace clusteridxdebug.csv] [--centroidtrace centroiddebug.csv]

Arguments:

 --input:
 
   Specifies input CSV file, number of rows represents number of points, the
   number of columns is the dimension of each point.

 --output:

   Output CSV file, just a single row, with as many entries as the number of
   points in the input file. Each entry is the index of the cluster to which
   the point belongs. The script 'visualize_clusters.py' can show this final
   clustering.

 --k:

   The number of clusters that should be identified.

 --repetitions:

   The number of times the k-means algorithm is repeated; the best clustering
   is kept.

 --blocks:

   Only relevant in CUDA version, specifies the number of blocks that can be
   used.

 --threads:

   Not relevant for the serial version. For the OpenMP version, this number 
   of threads should be used. For the CUDA version, this is the number of 
   threads per block. For the MPI executable, this should be ignored, but
   the wrapper script 'mpiwrapper.sh' can inspect this to run 'mpirun' with
   the correct number of processes.

 --seed:

   Specifies a seed for the random number generator, to be able to get 
   reproducible results.

 --trace:

   Debug option - do NOT use this when timing your program!

   For each repetition, the k-means algorithm goes through a sequence of 
   increasingly better cluster assignments. If this option is specified, this
   sequence of cluster assignments should be written to a CSV file, similar
   to the '--output' option. Instead of only having one line, there will be
   as many lines as steps in this sequence. If multiple repetitions are
   specified, only the results of the first repetition should be logged
   for clarity. The 'visualize_clusters.py' program can help to visualize
   the data logged in this file.

 --centroidtrace:

   Debug option - do NOT use this when timing your program!

   Should also only log data during the first repetition. The resulting CSV 
   file first logs the randomly chosen centroidIndices from the input data, and for
   each step in the sequence, the updated centroidIndices are logged. The program 
   'visualize_centroidIndices.py' can be used to visualize how the centroidIndices change.
   
)XYZ";
	exit(-1);
}

// Helper function to read input file into allData, setting number of detected
// rows and columns. Feel free to use, adapt or ignore
void readData(std::ifstream &input, std::vector<double> &allData, size_t &numRows, size_t &numCols)
{
	if (!input.is_open())
		throw std::runtime_error("Input file is not open");

	allData.resize(0);
	numRows = 0;
	numCols = -1;

	CSVReader inReader(input);
	int numColsExpected = -1;
	int line = 1;
	std::vector<double> row;

	while (inReader.read(row))
	{
		if (numColsExpected == -1)
		{
			numColsExpected = row.size();
			if (numColsExpected <= 0)
				throw std::runtime_error("Unexpected error: 0 columns");
		}
		else if (numColsExpected != (int)row.size())
			throw std::runtime_error("Incompatible number of colums read in line " + std::to_string(line) + ": expecting " + std::to_string(numColsExpected) + " but got " + std::to_string(row.size()));

		for (auto x : row)
			allData.push_back(x);

		line++;
	}

	numRows = (size_t)allData.size() / numColsExpected;
	numCols = (size_t)numColsExpected;
}

FileCSVWriter openDebugFile(const std::string &n)
{
	FileCSVWriter f;

	if (n.length() != 0)
	{
		f.open(n);
		if (!f.is_open())
			std::cerr << "WARNING: Unable to open debug file " << n << std::endl;
	}
	return f;
}

std::pair<size_t, double> find_closest_centroid_index_and_distance(std::vector<double> &allData, int pointIndex, std::vector<size_t> &centroidIndices, int numCols) {
    size_t closestCentroidIndex = 0;
    double closestDistance = std::numeric_limits<double>::max();

    for (size_t centroidIndex = 0; centroidIndex < centroidIndices.size(); ++centroidIndex) {
        double distance = 0;
        for (int dimensionIndex = 0; dimensionIndex < numCols; ++dimensionIndex) {
            distance += pow(allData[pointIndex * numCols + dimensionIndex] - allData[centroidIndices[centroidIndex] * numCols + dimensionIndex], 2);
        }
        distance = sqrt(distance);

        if (distance < closestDistance) {
            closestDistance = distance;
            closestCentroidIndex = centroidIndex;
        }
    }
    return std::make_pair(closestCentroidIndex, closestDistance);
}


int kmeans(Rng &rng, const std::string &inputFile, const std::string &outputFileName,
		   int numClusters, int repetitions, int numBlocks, int numThreads,
		   const std::string &centroidDebugFileName, const std::string &clusterDebugFileName)
{
	// If debug filenames are specified, this opens them. The is_open method
	// can be used to check if they are actually open and should be written to.
	FileCSVWriter centroidDebugFile = openDebugFile(centroidDebugFileName);
	FileCSVWriter clustersDebugFile = openDebugFile(clusterDebugFileName);

	FileCSVWriter csvOutputFile(outputFileName);
	if (!csvOutputFile.is_open())
	{
		std::cerr << "Unable to open output file " << outputFileName << std::endl;
		return -1;
	}

	// TODO: load dataset
	std::ifstream input(inputFile);
	if (!input.is_open())
	{
		std::cerr << "Unable to open input file " << inputFile << std::endl;
		return -1;
	}

	size_t numRows, numCols;
	std::vector<double> allData;
	readData(input, allData, numRows, numCols);

	// This is a basic timer from std::chrono ; feel free to use the appropriate timer for
	// each of the technologies, e.g. OpenMP has omp_get_wtime()
	Timer timer;

	double bestDistSquaredSum = std::numeric_limits<double>::max(); // can only get better
	std::vector<size_t> stepsPerRepetition(repetitions);			// to save the number of steps each rep needed

	// Do the k-means routine a number of times, each time starting from
	// different random centroidIndices (use Rng::pickRandomIndices), and keep
	// the best result of these repetitions.
	for (int r = 0; r < repetitions; r++)
	{
		size_t numSteps = 0;
		// TODO: perform an actual k-means run, starting from random centroidIndices
		//       (see rng.h)
		// std::cerr << "TODO: implement this" << std::endl;
		// First random centroid selection using given rng class
		std::vector<size_t> centroidIndices(numClusters);
		rng.pickRandomIndices(numRows, centroidIndices);
		std::vector<int> clusters(numRows, -1); // Initialize all points to no cluster
		// // std::cout << "Centroid Indices: " << std::endl;
		// // for (size_t i = 0; i < centroidIndices.size(); i++)
		// // {
		// // 	std::cout << centroidIndices[i] << std::endl;
		// // }

		// std::vector<size_t> clusters = std::vector<size_t>(numRows); // 2D punten -> index

		bool changed = true;

		// Loop until no changes are made
		while (changed)
		{
			// Set changed to false
			changed = false;
			double distanceSquaredSum = 0;
			// For each point in the dataset
			for (size_t pointIndex = 0; pointIndex < numRows; ++pointIndex)
			{
				std::pair<int, double> result = find_closest_centroid_index_and_distance(allData, pointIndex, centroidIndices, numCols);
				int newCluster = result.first;
				double dist = result.second;
				distanceSquaredSum += dist;

				if (newCluster != clusters[pointIndex]){
					clusters[pointIndex] = newCluster;
					changed = true;
				}
			}
			// If any of the points changed then recalculate the centroidIndices to the new average of the points in the cluster
			if (changed)
			{
				numSteps++;
				std::vector<std::vector<double>> newcentroidIndices(numClusters, std::vector<double>(numCols, 0));
                std::vector<int> clusterCounts(numClusters, 0);
				// Sum all dimensions and put in newCentroidIndices, later we devide this by count of points in that cluster
                for (size_t p = 0; p < numRows; ++p) {
                    int clusterIndex = clusters[p];
                    for (int dimensionIndex = 0; dimensionIndex < numCols; ++dimensionIndex) {
                        newcentroidIndices[clusterIndex][dimensionIndex] += allData[p * numCols + dimensionIndex];
                    }
                    clusterCounts[clusterIndex]++;
                }

                // Calculate the average, so we devided it with aantal points in that cluster
                for (int j = 0; j < numClusters; ++j) {
                    std::vector<double> newCentroid(numCols, 0);
                    for (int dimensionIndex = 0; dimensionIndex < numCols; ++dimensionIndex) {
                        if (clusterCounts[j] > 0) {
                            newCentroid[dimensionIndex] = newcentroidIndices[j][dimensionIndex] / clusterCounts[j];
                        }
                    }
                    
                    // We need to find the closest point to generated new centroid, so we calculate distances and get the shortest one as the new centroid.
                    double minDistance = std::numeric_limits<double>::max();
                    size_t closestPointIndex = 0;
                    for (size_t dataIndex = 0; dataIndex < numRows; ++dataIndex) {
                        double distance = 0;
                        for (int dimensionIndex = 0; dimensionIndex < numCols; ++dimensionIndex) {
                            double diff = allData[dataIndex * numCols + dimensionIndex] - newCentroid[dimensionIndex];
                            distance += diff * diff;
                        }
						distance = sqrt(distance);
                        if (distance < minDistance) {
                            minDistance = distance;
                            closestPointIndex = dataIndex;
                        }
                    }
                    
                    // Update centroidIndices and newcentroidIndices
                    centroidIndices[j] = closestPointIndex;
                }
			}

			// Update best clustering if needed
			if (distanceSquaredSum < bestDistSquaredSum) {
				bestDistSquaredSum = distanceSquaredSum;
				//bestClusters = clusters;
			}
		}
		
		stepsPerRepetition[r] = numSteps;

		// Make sure debug logging is only done on first iteration ; subsequent checks
		// with is_open will indicate that no logging needs to be done anymore.
		std::cout << "Centroid Indices (End): " << std::endl;
		std::string ownOutcome = "";
		for (size_t i = 0; i < clusters.size(); i++)
		{
			ownOutcome += std::to_string(clusters[i])+',';
		}
		std::string outcome = "0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,1,2,0,0,0,0,0,1,0,0,0,2,0,0,0,1,0,0,0,2,0,0,0,2,0,2,1,0,0,0,1,1,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,1,2,2,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,2,2,0,2,0,0,0,0,1,2,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,1,0,2,2,1,0,0,0,0,2,0,2,1,0,0,0,2,0,1,2,0,1,0,0,2,0,0,1,0,0,0,1,0,2,0,2,2,0,0,2,2,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,0,1,0,2,0,0,0,0,0,0,0,2,2,0,0,0,0,2,0,0,2,0,2,0,0,0,0,0,0,0,2,0,0,2,1,2,0,2,0,0,2,0,0,1,1,0,2,0,0,0,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0,2,2,0,0,2,0,0,0,0,0,2,0,0,0,2,0,2,0,0,0,0,2,0,0,0,0,2,2,2,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,1,0,0,1,2,2,2,1,";
		std::cout << "----------OwnOutcome-------------" << std::endl;
		std::cout << ownOutcome << std::endl;
		std::cout << "----------OutCome-------------" << std::endl;
		std::cout << outcome << std::endl;
		if (ownOutcome == outcome) {
			std::cout << "EQUAL" << std::endl;
		} else {
			std::cout << "NOT EQUAL" << std::endl;
		}
		centroidDebugFile.close();
		clustersDebugFile.close();
	}
	for (size_t i = 0; i < stepsPerRepetition.size(); i++)
	{
		std::cout << stepsPerRepetition[i] << " ";
	}
	std::cout << "-----------------------------" << std::endl;
	
	

	timer.stop();

	// Some example output, of course you can log your timing data anyway you like.
	std::cerr << "# Type,blocks,threads,file,seed,clusters,repetitions,bestdistsquared,timeinseconds" << std::endl;
	std::cout << "sequential," << numBlocks << "," << numThreads << "," << inputFile << ","
			  << rng.getUsedSeed() << "," << numClusters << ","
			  << repetitions << "," << bestDistSquaredSum << "," << timer.durationNanoSeconds() / 1e9
			  << std::endl;

	// Write the number of steps per repetition, kind of a signature of the work involved
	csvOutputFile.write(stepsPerRepetition, "# Steps: ");
	// Write best clusters to csvOutputFile, something like
	// csvOutputFile.write( best cluster indices )
	return 0;
}

int mainCxx(const std::vector<std::string> &args)
{
	if (args.size() % 2 != 0)
		usage();

	std::string inputFileName, outputFileName, centroidTraceFileName, clusterTraceFileName;
	unsigned long seed = 0;

	int numClusters = -1, repetitions = -1;
	int numBlocks = 1, numThreads = 1;
	for (int i = 0; i < args.size(); i += 2)
	{
		if (args[i] == "--input")
			inputFileName = args[i + 1];
		else if (args[i] == "--output")
			outputFileName = args[i + 1];
		else if (args[i] == "--centroidtrace")
			centroidTraceFileName = args[i + 1];
		else if (args[i] == "--trace")
			clusterTraceFileName = args[i + 1];
		else if (args[i] == "--k")
			numClusters = stoi(args[i + 1]);
		else if (args[i] == "--repetitions")
			repetitions = stoi(args[i + 1]);
		else if (args[i] == "--seed")
			seed = stoul(args[i + 1]);
		else if (args[i] == "--blocks")
			numBlocks = stoi(args[i + 1]);
		else if (args[i] == "--threads")
			numThreads = stoi(args[i + 1]);
		else
		{
			std::cerr << "Unknown argument '" << args[i] << "'" << std::endl;
			return -1;
		}
	}

	if (inputFileName.length() == 0 || outputFileName.length() == 0 || numClusters < 1 || repetitions < 1 || seed == 0)
		usage();

	Rng rng(seed);

	return kmeans(rng, inputFileName, outputFileName, numClusters, repetitions,
				  numBlocks, numThreads, centroidTraceFileName, clusterTraceFileName);
}

int main(int argc, char *argv[])
{
	std::vector<std::string> args;
	for (int i = 1; i < argc; i++)
		args.push_back(argv[i]);

	return mainCxx(args);
}

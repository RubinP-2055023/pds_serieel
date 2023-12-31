#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include "CSVReader.hpp"
#include "CSVWriter.hpp"
#include "rng.h"
#include "timer.h"
#include <omp.h>

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

void generateCentroidsUsingRng(Rng &rng, const std::vector<double> &allData, std::vector<double> &centroids, int numClusters, size_t numRows, size_t numCols)
{
    std::vector<size_t> centroidIndices(numClusters);
    rng.pickRandomIndices(numRows, centroidIndices);

    // Read the centroids point from data and put in centroids vector.
    for (size_t centroidIndex = 0; centroidIndex < centroidIndices.size(); centroidIndex++)
    {
        for (size_t dimensionIndex = 0; dimensionIndex < numCols; dimensionIndex++)
        {
            centroids[centroidIndex * numCols + dimensionIndex] = allData[centroidIndices[centroidIndex] * numCols + dimensionIndex];
        }
    }
}

void findClosestCentroidIndexAndDistance(const std::vector<double> &allData, size_t pointIndex,
                                         const std::vector<double> &centroids,
                                         size_t &closestCentroidIndex, double &closestDistance, size_t numCols)
{
    closestCentroidIndex = 0;
    closestDistance = std::numeric_limits<double>::max();
    int countCentroids = centroids.size() / numCols;
    
    for (size_t centroidIndex = 0; centroidIndex < countCentroids; centroidIndex++)
    {
        double distance = 0;
        for (size_t dimensionIndex = 0; dimensionIndex < numCols; ++dimensionIndex)
        {
            double diff = allData[pointIndex * numCols + dimensionIndex] - centroids[centroidIndex * numCols + dimensionIndex];
            distance += diff * diff;
        }
        if (distance < closestDistance)
        {
            closestDistance = distance;
            closestCentroidIndex = centroidIndex;
        }
    }
}

std::vector<double> calculateNewCentroid(const std::vector<double> &allData, const std::vector<int> &clusters, size_t clusterIndex, size_t numCols)
{
    std::vector<double> newCentroid(numCols, 0);
    size_t numPoints = 0;
    for (size_t pointIndex = 0; pointIndex < clusters.size(); ++pointIndex)
    {
        if (clusters[pointIndex] == clusterIndex)
        {
            for (size_t dimensionIndex = 0; dimensionIndex < numCols; ++dimensionIndex)
            {
                newCentroid[dimensionIndex] += allData[pointIndex * numCols + dimensionIndex];
            }
            ++numPoints;
        }
    }

    for (size_t dimensionIndex = 0; dimensionIndex < numCols; ++dimensionIndex)
    {
        newCentroid[dimensionIndex] /= numPoints;
    }
    return newCentroid;
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

    // Initialize the best clusters and distance sum
    std::vector<size_t> bestClusters(numRows); // To store the cluster assignments
    double bestDistanceSquaredSum = std::numeric_limits<double>::max();
    std::vector<int> stepsPerRepetition(repetitions, 0);

    std::vector<double> centroidsHistory;


    Timer timer(false);
    double omp_start_time;
    if(numThreads)
    {
        omp_start_time = omp_get_wtime();
    }
    else
    {
        timer.start();
    }

    // Main k-means loop
    for (int r = 0; r < repetitions; r++)
    {

        std::vector<double> centroids(numClusters * numCols);
        // Step 1: Initialize oudeCentroids by randomly choosing k points
        generateCentroidsUsingRng(rng, allData, centroids, numClusters, numRows, numCols);
        
        if (r == 0)
        {
            for (size_t j = 0; j < numClusters; j++)
            {
                for (size_t dimensionIndex = 0; dimensionIndex < numCols; ++dimensionIndex)
                {
                    centroidsHistory.push_back(centroids[j * numCols + dimensionIndex]);
                }
            }
        }
        // Initialize cluster assignments
        std::vector<int> clusters(numRows, -1); // Initially, all points are unassigned to clusters

        // Flags for tracking changes in clustering
        bool changed = true;
        double distanceSquaredSum; // Declare outside the while loop
        size_t numSteps = 0;
        // Main k-means loop
        while (changed)
        {
            changed = false;
            distanceSquaredSum = 0.0; // Initialize distance squared sum

            // Step 2: Assign each point to the nearest centroid
            #pragma omp parallel for num_threads(numThreads) reduction(+:distanceSquaredSum)
            for (size_t p = 0; p < numRows; p++)
            {
                size_t newCluster;
                double dist;
                findClosestCentroidIndexAndDistance(allData, p, centroids, newCluster, dist, numCols);
                distanceSquaredSum += dist;
                if (newCluster != clusters[p])
                {
                    clusters[p] = newCluster;
                    changed = true;
                }
            }

            // Step 3: Recalculate oudeCentroids based on current clustering
            #pragma omp parallel for num_threads(numThreads)
            for (size_t j = 0; j < numClusters; j++)
            {
                std::vector<double> newCentroid = calculateNewCentroid(allData, clusters, j, numCols);
                for (size_t dimensionIndex = 0; dimensionIndex < numCols; ++dimensionIndex)
                {
                    centroids[j * numCols + dimensionIndex] = newCentroid[dimensionIndex];
                }
            }
            ++numSteps;
            if (r == 0)
            {
                for (size_t j = 0; j < numClusters; j++)
                {
                    for (size_t dimensionIndex = 0; dimensionIndex < numCols; ++dimensionIndex)
                    {
                        centroidsHistory.push_back(centroids[j * numCols + dimensionIndex]);
                    }
                }
                if (clustersDebugFile.is_open()) 
                    clustersDebugFile.write(clusters);
                }
        }
        // Keep track of the number of steps per repetition
        stepsPerRepetition[r] = numSteps;
        if (distanceSquaredSum < bestDistanceSquaredSum)
        {
            bestClusters.assign(clusters.begin(), clusters.end());
            bestDistanceSquaredSum = distanceSquaredSum;
        }

    }

    double elapsed;
    if(numThreads)
    {
        elapsed = omp_get_wtime() - omp_start_time;
    }
    else
    {
        timer.stop();
        elapsed = timer.durationNanoSeconds() / 1e9;
    }

    // Some example output, of course you can log your timing data anyway you like.
    std::cerr << "# Type,blocks,threads,file,seed,clusters,repetitions,bestdistsquared,timeinseconds" << std::endl;
    std::cout << "sequential," << numBlocks << "," << numThreads << "," << inputFile << ","
              << rng.getUsedSeed() << "," << numClusters << ","
              << repetitions << "," << bestDistanceSquaredSum << "," << elapsed
              << std::endl;

    // Write the number of steps per repetition, kind of a signature of the work involved
    std::string linePrefix = "";

    csvOutputFile.write(stepsPerRepetition, "# Steps: ");
    csvOutputFile.write(bestClusters);
    csvOutputFile.close();

    if (numClusters*numCols <= centroidsHistory.size()) {
        centroidsHistory.erase(centroidsHistory.end() - numClusters*numCols, centroidsHistory.end());
    }

    if (centroidDebugFile.is_open()){
        centroidDebugFile.write(centroidsHistory, numCols,linePrefix);
        centroidDebugFile.close();
    }

    if (clustersDebugFile.is_open()){
        //clustersDebugFile.write();
        clustersDebugFile.close();
    }

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

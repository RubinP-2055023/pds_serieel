#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include "CSVReader.hpp"
#include "CSVWriter.hpp"
#include "rng.h"
#include "timer.h"
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
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

__global__ void findClosestCentroidKernel(const double *allData, const double *centroids,
                                          int *clusters, bool *changes, double *distanceSquaredSum,
                                          size_t numCols, size_t numRows, int numClusters, int loopSize, int offset)
{
    // Calculate newCluster and set the flag for changes
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = 0; i < loopSize; i++)
    {

        int index = tid + i * offset;
        if (index < numRows)
        {
            double closestDistance = 1e9;
            size_t closestCentroidIndex = 0;

            for (size_t centroidIndex = 0; centroidIndex < numClusters; centroidIndex++)
            {
                double distance = 0;
                for (size_t dimensionIndex = 0; dimensionIndex < numCols; ++dimensionIndex)
                {
                    double diff = allData[index * numCols + dimensionIndex] - centroids[centroidIndex * numCols + dimensionIndex];
                    distance += diff * diff;
                }

                if (distance < closestDistance)
                {
                    closestDistance = distance;
                    closestCentroidIndex = centroidIndex;
                }
            }

            if (closestCentroidIndex != clusters[index])
            {
                clusters[index] = closestCentroidIndex;
                changes[index] = true;
            }

            atomicAdd(distanceSquaredSum, closestDistance);
        }
    }
}

void findClosestCentroidCUDA(const double *allData, const double *centroids,
                             int *clusters, thrust::device_vector<bool> &changes, double &distanceSquaredSum, size_t numCols, size_t numRows, int numClusters, int numThreads, int numBlocks)
{
    // Allocate device memory for data
    thrust::device_vector<double> d_allData(allData, allData + numRows * numCols);
    thrust::device_vector<double> d_centroids(centroids, centroids + numClusters * numCols);
    thrust::device_vector<int> d_clusters(clusters, clusters + numRows);

    double *d_distanceSquaredSum;
    cudaMalloc(&d_distanceSquaredSum, sizeof(double));
    cudaMemcpy(d_distanceSquaredSum, &distanceSquaredSum, sizeof(double), cudaMemcpyHostToDevice);

    // Launch CUDA kernel

    int threads = numThreads;
    int loopSize = (numRows + threads * numBlocks - 1) / (threads * numBlocks);
    int offset = threads * numBlocks;

    findClosestCentroidKernel<<<numBlocks, threads>>>(thrust::raw_pointer_cast(d_allData.data()),
                                                      thrust::raw_pointer_cast(d_centroids.data()),
                                                      thrust::raw_pointer_cast(d_clusters.data()),
                                                      thrust::raw_pointer_cast(changes.data()),
                                                      d_distanceSquaredSum,
                                                      numCols, numRows, numClusters, loopSize, offset);

    // Synchronize the device to ensure the kernel completes before returning
    cudaDeviceSynchronize();

    // Copy results back to host if needed
    thrust::copy(d_clusters.begin(), d_clusters.end(), clusters);
    // No need to copy changes back to host if you don't need it on the host
    // Copy the updated distanceSquaredSum back to host
    cudaMemcpy(&distanceSquaredSum, d_distanceSquaredSum, sizeof(double), cudaMemcpyDeviceToHost);

    // Free allocated device memory
    cudaFree(d_distanceSquaredSum);
}

__global__ void accumulateCentroidsAndCountPointsKernel(const double *allData, double *newCentroids, int *clusterNumPoints,
                                                        const int *clusters, size_t numCols, size_t numPoints, size_t numClusters, int loopSize, int offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < loopSize; i++)
    {
        int index = tid + i * offset;

        if (index < numPoints)
        {
            int clusterIndex = clusters[index];
            // Accumulate values for new centroids
            for (size_t dimensionIndex = 0; dimensionIndex < numCols; ++dimensionIndex)
            {
                atomicAdd(&newCentroids[clusterIndex * numCols + dimensionIndex], allData[index * numCols + dimensionIndex]);
            }
            // Increment the count of points associated with the cluster
            atomicAdd(&clusterNumPoints[clusterIndex], 1);
        }
    }
}

// Fix kernel launch parameters
__global__ void normalizeCentroidsKernel(double *newCentroids, const int *clusterNumPoints, size_t numCols, size_t numClusters, int loopSize, int offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < loopSize; i++)
    {
        int index = tid + i * offset;

        if (index < numClusters)
        {
            for (size_t dimensionIndex = 0; dimensionIndex < numCols; ++dimensionIndex)
            {
                newCentroids[index * numCols + dimensionIndex] /= clusterNumPoints[index];
            }
        }
    }
}
void calculateNewCentroidCUDA(const double *allData, double *centroids, const int *clusters, size_t numCols, size_t numRows, int numClusters, int numThreads, int numBlocks)
{
    // Allocate device memory for data
    thrust::device_vector<double> d_allData(allData, allData + numRows * numCols);
    thrust::device_vector<double> d_centroids(centroids, centroids + numClusters * numCols);
    thrust::device_vector<int> d_clusters(clusters, clusters + numRows);
    std::vector<double> newCentroid(numCols * numClusters, 0);
    thrust::device_vector<double> d_newCentroids = newCentroid;
    thrust::device_vector<int> d_clusterNumPoints(numClusters, 0);

    int threads = numThreads;
    int loopSize = (numRows + threads * numBlocks - 1) / (threads * numBlocks);
    int offset = threads * numBlocks;

    accumulateCentroidsAndCountPointsKernel<<<numBlocks, threads>>>(thrust::raw_pointer_cast(d_allData.data()),
                                                                      thrust::raw_pointer_cast(d_newCentroids.data()),
                                                                      thrust::raw_pointer_cast(d_clusterNumPoints.data()),
                                                                      thrust::raw_pointer_cast(d_clusters.data()),
                                                                      numCols, numRows, numClusters, loopSize, offset);
    cudaDeviceSynchronize();
    
    loopSize = (numClusters + threads * numBlocks - 1) / (threads * numBlocks);

    normalizeCentroidsKernel<<<numBlocks, threads>>>(thrust::raw_pointer_cast(d_newCentroids.data()),
                                                                                       thrust::raw_pointer_cast(d_clusterNumPoints.data()),
                                                                                       numCols, numClusters, loopSize, offset);
    cudaDeviceSynchronize();
    thrust::copy(d_newCentroids.begin(), d_newCentroids.end(), centroids);
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
    Timer timer;
    // Main k-means loop
    // Initialize changes vector
    thrust::device_vector<bool> changes(numRows, false);
    for (int r = 0; r < repetitions; r++)
    {

        std::vector<double> centroids(numClusters * numCols);
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
            // Calculate
            {
                changed = false;
                distanceSquaredSum = 0.0;
                thrust::fill(changes.begin(), changes.end(), false);
                findClosestCentroidCUDA(allData.data(), centroids.data(), clusters.data(), changes, distanceSquaredSum, numCols, numRows, numClusters, numThreads, numBlocks);
                changed = thrust::count(changes.begin(), changes.end(), true) > 0;
                cudaDeviceSynchronize();
            }
            {
                calculateNewCentroidCUDA(allData.data(), centroids.data(), clusters.data(), numCols, numRows, numClusters, numThreads, numBlocks);
                cudaDeviceSynchronize();
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
            std::cout << "Best r: " << r << " with " << numSteps << " steps." << std::endl;
            bestClusters.assign(clusters.begin(), clusters.end());
            bestDistanceSquaredSum = distanceSquaredSum;
        }
    }

    timer.stop();

    // Some example output, of course you can log your timing data anyway you like.
    std::cerr << "# Type,blocks,threads,file,seed,clusters,repetitions,bestdistsquared,timeinseconds" << std::endl;
    std::cout << "sequential," << numBlocks << "," << numThreads << "," << inputFile << ","
              << rng.getUsedSeed() << "," << numClusters << ","
              << repetitions << "," << bestDistanceSquaredSum << "," << timer.durationNanoSeconds() / 1e9
              << std::endl;

    // Write the number of steps per repetition, kind of a signature of the work involved
    std::string linePrefix = "";

    csvOutputFile.write(stepsPerRepetition, "# Steps: ");
    csvOutputFile.write(bestClusters);
    csvOutputFile.close();

    // Deletes last k centroids because they are the same as k before them.
    if (numClusters * numCols <= centroidsHistory.size())
    {
        centroidsHistory.erase(centroidsHistory.end() - numClusters * numCols, centroidsHistory.end());
    }

    if (centroidDebugFile.is_open())
    {
        centroidDebugFile.write(centroidsHistory, numCols, linePrefix);
        centroidDebugFile.close();
    }

    if (clustersDebugFile.is_open())
    {
        // clustersDebugFile.write();
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
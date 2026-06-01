#include <vector>
#include <string>
#include <stdexcept>
#include <ctime>
#include <cstdlib>
#include <memory>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <cmath>
#include <limits>
#include <thread>
#include <iomanip>
#include <chrono>
#include <random>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

using namespace std;

#define EPS (1e-9)

// CUDA Error Checking Helper
#define gpu_error_check(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

#define STB_IMAGE_IMPLEMENTATION
#define IMAGE_SIDE 512
#define IMAGE_SIZE (IMAGE_SIDE * IMAGE_SIDE)
#include "stb_image.h"

// Evolutionary Algorithm Configuration
#define POPULATION_SIZE 1000
#define TOURNAMENT_SIZE 5
#define MAX_DURATION_SECONDS 600
#define EXPLORATION_RATE 15 // percentage chance of big mutation
#define MUTATION_RANGE 10 // fixed mutation range (for small mutations)

// GPU Configuration
#define BLOCK_SIZE 256

// Logging Settings
#define LOG_INTERVAL 500
#define SAVE_TOP_N 5 // save top 5 results
string fitness_log_file = "fitness_log.csv";

// ASCII Settings
static const string ASCII_RAMP = " .`'^\",:;Il!i~+_-?][}{1)(|\\/*tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
#define ASCII_COLS 256
#define ASCII_BLOCK_W (IMAGE_SIDE / ASCII_COLS)
#define ASCII_BLOCK_H ASCII_BLOCK_W
#define RAMP_SIZE 69
#define GENOME_LENGTH (ASCII_COLS * (IMAGE_SIDE / ASCII_BLOCK_H))

// GPU Constant Memory for lookups
__constant__ double D_RAMP_GREYNESS[RAMP_SIZE];

// logs epoch statistics to a CSV file
void save_epoch_stats(int epoch, double best_fitness, double avg_fitness, const string& log_path) {
    ofstream log_file(log_path, ios::app);
    if (log_file.is_open()) {
        if (epoch == 0) {
            log_file << "Epoch,Best_Fitness,Avg_Fitness" << endl;
        }
        log_file << epoch + 1 << "," << best_fitness << "," << avg_fitness << endl;
    }
}

// loads image and get greyscale values
vector<unsigned char> load_image(const string& path) {
    int x, y, channels;
    unsigned char* data = stbi_load(path.c_str(), &x, &y, &channels, 4);
    if (!data) throw runtime_error("Failed to load image: " + path);
    if (x != IMAGE_SIDE || y != IMAGE_SIDE) {
        stbi_image_free(data);
        throw runtime_error("Image size must be 512x512");
    }
    // convert RGB values to greyscale using the formula (0.299*R + 0.587*G + 0.114*B)
    vector<unsigned char> grey(IMAGE_SIZE);
    for (int i = 0; i < IMAGE_SIZE; i++) {
        int r = data[4 * i + 0];
        int g = data[4 * i + 1];
        int b = data[4 * i + 2];
        grey[i] = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);
    }
    stbi_image_free(data);
    return grey;
}

// computes average greyscale values for each ASCII block in the target image
vector<double> compute_target_block_averages(const vector<unsigned char>& grey) {
    const int blocks_in_row = ASCII_COLS;
    const int blocks_in_col = IMAGE_SIDE / ASCII_BLOCK_H;
    vector<double> avgs(blocks_in_row * blocks_in_col, 0.0);
    // for each block ...
    for (int by = 0; by < blocks_in_col; by++) {
        for (int bx = 0; bx < blocks_in_row; bx++) {
            // ... compute the sum of greyscale values ...
            long long sum = 0;
            for (int y = 0; y < ASCII_BLOCK_H; y++) {
                int gy = by * ASCII_BLOCK_H + y;
                int row_off = gy * IMAGE_SIDE;
                for (int x = 0; x < ASCII_BLOCK_W; x++) {
                    int gx = bx * ASCII_BLOCK_W + x;
                    sum += grey[row_off + gx];
                }
            }
            // ... and divide by number of pixels to get average
            avgs[by * blocks_in_row + bx] = static_cast<double>(sum) / (ASCII_BLOCK_W * ASCII_BLOCK_H);
        }
    }
    return avgs;
}

// CUDA Kernel to initialize population with random values
__global__ void init_population_kernel(unsigned char* population, unsigned long seed) {
    int individual_ind = blockIdx.x;
    int thread_id = threadIdx.x;
    if (individual_ind < POPULATION_SIZE) {
        int start_ind = individual_ind * GENOME_LENGTH;
        for (int i = thread_id; i < GENOME_LENGTH; i += blockDim.x) {
            // LCG (Linear Congruential Generator) to generate pseudo-random numbers - hash function for ASCII chars
            // method taken from the internet - ref [5] (report)
            unsigned int hash = (start_ind + i) * 1664525u + 1013904223u + seed;
            population[start_ind + i] = (unsigned char)((hash ^ (hash >> 5)) % RAMP_SIZE);
        }
    }
}

// kernel to evaluate individuals by computing SSE and Fitness
__global__ void evaluate_kernel(const unsigned char* population, double* fitness_values, double* sse_values, const double* target_avgs) {
    int individual_index = blockIdx.x;
    int thread_id = threadIdx.x;

    if (individual_index < POPULATION_SIZE) {
        int start_ind = individual_index * GENOME_LENGTH;
        double partial_sse = 0.0;

        // each thread computes partial SSE for its assigned genes
        for (int i = thread_id; i < GENOME_LENGTH; i += blockDim.x) {
            unsigned char char_ind = population[start_ind + i];
            double g = D_RAMP_GREYNESS[char_ind];
            double t = target_avgs[i];
            double diff = g - t;
            partial_sse += diff * diff;
        }

        // array shared between threads to combine partial SSEs
        __shared__ double shared_data[BLOCK_SIZE];
        shared_data[thread_id] = partial_sse;
        __syncthreads();

        // parallel reduction to sum up partial SSEs in logarithmic time
        // method taken from the internet - ref [3] (report)
        if (BLOCK_SIZE >= 256) {
            if (thread_id < 128) {
                shared_data[thread_id] += shared_data[thread_id + 128];
            }
            __syncthreads();
        }
        if (BLOCK_SIZE >= 128) {
            if (thread_id < 64) {
                shared_data[thread_id] += shared_data[thread_id + 64];
            }
            __syncthreads();
        }
        if (thread_id < 32) {
            // it is necessary to read volatile variable from memory
            // to prevent compiler optimization and old data from shared_data
            volatile double* volotile_mem = shared_data;
            volotile_mem[thread_id] += volotile_mem[thread_id + 32];
            volotile_mem[thread_id] += volotile_mem[thread_id + 16];
            volotile_mem[thread_id] += volotile_mem[thread_id + 8];
            volotile_mem[thread_id] += volotile_mem[thread_id + 4];
            volotile_mem[thread_id] += volotile_mem[thread_id + 2];
            volotile_mem[thread_id] += volotile_mem[thread_id + 1];
        }

        if (thread_id == 0) {
            double total_sse = shared_data[0];
            sse_values[individual_index] = total_sse;
            double mse = total_sse / (double)GENOME_LENGTH;
            double linear_fit = 1.0 / (mse + EPS);
            fitness_values[individual_index] = linear_fit;
        }
    }
}

// algorithm for pseudo-random numbers generation
// taken from the internet - ref [2] (report)
// replaces curand for memory usage optimization
__device__ inline unsigned int xorshift32(unsigned int* seed) {
    unsigned int x = *seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *seed = x;
    return x;
}

// kernel to copy elite individuals to the next generation
__global__ void elitism_kernel(const int* sorted_indices, const unsigned char* source, unsigned char* destination, int n_elites, int genome_len) {
    int individual_ind = blockIdx.x;
    if (individual_ind < n_elites) {
        int thread_id = threadIdx.x;
        int best_ind = sorted_indices[individual_ind];
        int start_destination = individual_ind * genome_len;
        int start_source = best_ind * genome_len;
        for (int i = thread_id; i < genome_len; i += blockDim.x) {
            destination[start_destination + i] = source[start_source + i];
        }
    }
}

// kernel to generate the next population via crossover and mutation
__global__ void generation_kernel(const unsigned char* old_population, unsigned char* new_population, const double* fitness_values, int elites_count, unsigned long seed) {
    int individual_ind = blockIdx.x;
    int thread_id = threadIdx.x;

    __shared__ int p1_ind;
    __shared__ int p2_ind;
    __shared__ unsigned int rng_seed;

    if (individual_ind >= elites_count && individual_ind < POPULATION_SIZE) {
        if (thread_id == 0) {
            rng_seed = (unsigned int)(seed + individual_ind * 54321);

            // tournament selection for parent 1
            int best = xorshift32(&rng_seed) % POPULATION_SIZE;
            for (int k = 0; k < TOURNAMENT_SIZE - 1; k++) {
                int chal = xorshift32(&rng_seed) % POPULATION_SIZE;
                if (fitness_values[chal] > fitness_values[best]) best = chal;
            }
            p1_ind = best;

            // tournament selection for parent 2
            best = xorshift32(&rng_seed) % POPULATION_SIZE;
            for (int k = 0; k < TOURNAMENT_SIZE - 1; k++) {
                int chal = xorshift32(&rng_seed) % POPULATION_SIZE;
                if (fitness_values[chal] > fitness_values[best]) best = chal;
            }
            p2_ind = best;
        }
        __syncthreads();

        int child_start = individual_ind * GENOME_LENGTH;
        int p1_start = p1_ind * GENOME_LENGTH;
        int p2_start = p2_ind * GENOME_LENGTH;

        // Crossover
        for (int i = thread_id; i < GENOME_LENGTH; i += blockDim.x) {
            // rnadom number generation for crossover decision
            unsigned int decision = (i * 12345u + individual_ind * 67890u + seed);
            // select gene from parent 1 or parent 2 based on decision LSB
            new_population[child_start + i] = (decision & 1) ? old_population[p1_start + i] : old_population[p2_start + i];
        }
        __syncthreads();

        // Mutation
        if (thread_id == 0) {
            int mut_ind = xorshift32(&rng_seed) % GENOME_LENGTH;
            unsigned char old_char = new_population[child_start + mut_ind];
            unsigned char new_char;

            if ((xorshift32(&rng_seed) % 100) < EXPLORATION_RATE) {
                // Big mutation (Exploration) - to any possible character
                new_char = (unsigned char)(xorshift32(&rng_seed) % RAMP_SIZE);
            }
            else {
                // Small mutation (Exploitation) - in a predefined step range
                int step = (xorshift32(&rng_seed) % (2 * MUTATION_RANGE + 1)) - MUTATION_RANGE;
                int next_val = (int)old_char + step;
                if (next_val < 0) next_val = 0;
                if (next_val >= RAMP_SIZE) next_val = RAMP_SIZE - 1;
                new_char = (unsigned char)next_val;
            }
            new_population[child_start + mut_ind] = new_char;
        }
    }
}

// saves the genome as an ASCII text file
void save_ascii_txt(const string& filename, const unsigned char* genome_data) {
    ofstream out(filename);
    int cols = ASCII_COLS;
    int max_ind = RAMP_SIZE - 1;

    for (int i = 0; i < GENOME_LENGTH; i++) {
        int char_ind = static_cast<int>(genome_data[i]);
        out << ASCII_RAMP[char_ind];
        if ((i + 1) % cols == 0) {
            out << '\n';
        }
    }
}

// creates a string in a format needed to write a report
void make_report_string(string& str, int image_ind, int run_ind) {
    str += to_string(image_ind) + "_" + to_string(run_ind);
}

// executes the evolutionary algorithm
// h for host variables (accessible by CPU)
// d for device variables (accessible by GPU)
void execute(const string& input_path, string log_path, int image_ind = 0, int run_ind = 0) {
    if (!filesystem::exists(input_path)) {
        cout << "File not found: " << input_path << endl;
        exit(1);
    }

    string full_log_path = log_path + "_" + to_string(image_ind) + "_" + to_string(run_ind) + ".csv";

    vector<unsigned char> image = load_image(input_path);
    vector<double> h_target_avgs = compute_target_block_averages(image);

    vector<double> h_ramp_greyness(RAMP_SIZE);
    for (int i = 0; i < RAMP_SIZE; i++) {
        h_ramp_greyness[i] = (static_cast<double>(i) * 255.0) / (RAMP_SIZE - 1);
    }

    // allocate GPU Memory
    unsigned char* d_population_parents, * d_population_children;
    double* d_fitness, * d_sse;
    int* d_indices;
    double* d_target_avgs;
    double* d_fitness_copy;

    size_t poplation_mem_size = (size_t)POPULATION_SIZE * (size_t)GENOME_LENGTH * sizeof(unsigned char);

    gpu_error_check(cudaMalloc(&d_population_parents, poplation_mem_size));
    gpu_error_check(cudaMalloc(&d_population_children, poplation_mem_size));
    gpu_error_check(cudaMalloc(&d_fitness, POPULATION_SIZE * sizeof(double)));
    gpu_error_check(cudaMalloc(&d_sse, POPULATION_SIZE * sizeof(double)));
    gpu_error_check(cudaMalloc(&d_indices, POPULATION_SIZE * sizeof(int)));
    gpu_error_check(cudaMalloc(&d_target_avgs, GENOME_LENGTH * sizeof(double)));
    gpu_error_check(cudaMalloc(&d_fitness_copy, POPULATION_SIZE * sizeof(double)));

    gpu_error_check(cudaMemcpyToSymbol(D_RAMP_GREYNESS, h_ramp_greyness.data(), RAMP_SIZE * sizeof(double)));
    gpu_error_check(cudaMemcpy(d_target_avgs, h_target_avgs.data(), GENOME_LENGTH * sizeof(double), cudaMemcpyHostToDevice));

    // initialize Population with random values
    init_population_kernel <<<POPULATION_SIZE, BLOCK_SIZE>>> (d_population_parents, time(NULL));

    unsigned char* d_current_population = d_population_parents;
    unsigned char* d_next_population = d_population_children;

    // creating GPU pointers
    thrust::device_ptr<double> t_fitness(d_fitness);
    thrust::device_ptr<int> t_indices(d_indices);

    cout << "STARTING EVOLUTIONARY ALGORITHM\n";

    auto start_time = chrono::steady_clock::now();
    int epoch = 0;
    bool running = true;

    while (running) {
        evaluate_kernel <<<POPULATION_SIZE, BLOCK_SIZE>>> (d_current_population, d_fitness, d_sse, d_target_avgs);
        // copy fintess values for generation step
        gpu_error_check(cudaMemcpy(d_fitness_copy, d_fitness, POPULATION_SIZE * sizeof(double), cudaMemcpyDeviceToDevice));
        // sorting by fitness
        thrust::sequence(t_indices, t_indices + POPULATION_SIZE);
        thrust::sort_by_key(t_fitness, t_fitness + POPULATION_SIZE, t_indices, thrust::greater<double>());
        int elites = SAVE_TOP_N;
        elitism_kernel <<<elites, BLOCK_SIZE>>> (d_indices, d_current_population, d_next_population, elites, GENOME_LENGTH);

        if (epoch % LOG_INTERVAL == 0) {
            double current_best;
            cudaMemcpy(&current_best, d_fitness, sizeof(double), cudaMemcpyDeviceToHost);

            // calculate average fitness across the population using thrust::reduce
            double sum_fitness = thrust::reduce(t_fitness, t_fitness + POPULATION_SIZE, 0.0, thrust::plus<double>());
            double avg_fitness = sum_fitness / POPULATION_SIZE;

            // check time and save stats
            auto now = chrono::steady_clock::now();
            double elapsed = chrono::duration_cast<chrono::seconds>(now - start_time).count();
            cout << "Epoch " << epoch << " | Best: " << current_best << " | Avg: " << avg_fitness << " | Time: " << elapsed << "s" << endl;
            save_epoch_stats(epoch, current_best, avg_fitness, full_log_path);
            if (elapsed >= MAX_DURATION_SECONDS) {
                cout << ">> TIME LIMIT REACHED (" << elapsed << "s)" << endl;
                running = false;
            }
        }
        // break if time limit is reached
        if (!running) break;
        generation_kernel <<<POPULATION_SIZE, BLOCK_SIZE>>> (d_current_population, d_next_population, d_fitness_copy, elites, time(NULL) + epoch * 999);
        swap(d_current_population, d_next_population);
        epoch++;
    }

    // final sort to find the best individuals
    evaluate_kernel <<<POPULATION_SIZE, BLOCK_SIZE>>> (d_current_population, d_fitness, d_sse, d_target_avgs);
    thrust::sequence(t_indices, t_indices + POPULATION_SIZE);
    thrust::sort_by_key(t_fitness, t_fitness + POPULATION_SIZE, t_indices, thrust::greater<double>());
    vector<int> h_best_indices(SAVE_TOP_N);
    cudaMemcpy(h_best_indices.data(), d_indices, SAVE_TOP_N * sizeof(int), cudaMemcpyDeviceToHost);

    // saving best results
    string results = "results";
    make_report_string(results, image_ind, run_ind);
    filesystem::create_directories(results);

    for (int i = 0; i < SAVE_TOP_N; i++) {
        int best_ind = h_best_indices[i];
        vector<unsigned char> genome(GENOME_LENGTH);
        cudaMemcpy(genome.data(), d_current_population + (best_ind * GENOME_LENGTH), GENOME_LENGTH, cudaMemcpyDeviceToHost);

        string filename = "LeonidGunkoOutput";
        make_report_string(filename, image_ind, run_ind);
        filename += "_" + to_string(i + 1) + ".txt";
        string save_path = results + "/" + filename;
        save_ascii_txt(save_path, genome.data());
    }

    cudaFree(d_population_parents);
    cudaFree(d_population_children);
    cudaFree(d_fitness);
    cudaFree(d_sse);
    cudaFree(d_indices);
    cudaFree(d_fitness_copy);
    cudaFree(d_target_avgs);
}

int main() {
    srand(static_cast<unsigned int>(time(NULL)));
    cout << fixed;
    cout.precision(10);

    /* FOR REPORT STATISTICS COLLECTION
    for (int x = 1; x <= 5; x++) {
        string input_image = "tests/input" + to_string(x) + ".jpg";
        for (int y = 1; y <= 3; y++) {
            cout << "Execution for image " << x << "; run " << y << "\n";
            execute(input_image, "log", x, y);
        }
    }
    */

    string input_image = "tests/test.jpg";
    execute(input_image, "log", 0, 0);
}
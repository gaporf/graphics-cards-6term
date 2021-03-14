#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include <cstdio>
#include <chrono>

#include <omp.h>

#include "pgm_image.h"

size_t get_threads(int32_t threads) {
	if (threads == -1) return 1;
	if (threads == 0) return omp_get_max_threads();
	return threads;
}

int main(int argc, char* argv[]) {
	if (argc != 4) {
		std::cerr << "Input format: <input file> <output file> <num of threads>" << std::endl;
		return 1;
	}
	int32_t threads;
	{
		try {
			size_t n;
			threads = std::stoi(argv[3], &n);
			if (argv[3][n] != '\0') throw std::runtime_error("Expected number");
		} catch (...) {
			std::cerr << "Expected the number as a third argument" << std::endl;
			return 1;
		}
		if (threads < -1) {
			std::cerr << "Expected the number not less than -1" << std::endl;
			return 1;
		}
	}
	uint32_t buf[256];
	try {
		pgm_image image(argv[1]);
		auto t_start = std::chrono::high_resolution_clock::now();
		if (threads == -1) {
			image.get_histogram(buf);
		} else {
			image.get_histogram(buf, threads);
		}
		auto t_end = std::chrono::high_resolution_clock::now();
		std::cout << std::endl << "Time (" << get_threads(threads) << " thread(s)): " << std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms" << std::endl;
	} catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}
	std::ofstream output(argv[2], std::ios_base::binary);
	if (output.fail()) {
		std::cerr << "Could not open file to write" << std::endl;
		return 1;
	}
	output.write(reinterpret_cast<char*>(&buf), 1024);
	if (output.fail()) {
		std::cerr << "Could not open file to write" << std::endl;
		std::remove(argv[2]);
		return 1;
	}
	return 0;
}
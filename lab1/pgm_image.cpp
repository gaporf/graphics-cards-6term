#include <fstream>
#include <exception>

#include <omp.h>

#include "pgm_image.h"

static uint32_t get_number(char const* s) {
	char const* cur = s;
	if (*cur == '\0') throw std::runtime_error("Incorrect format of file");
	while (*cur != '\0') {
		if (!std::isdigit(*cur)) throw std::runtime_error("Incorrect format of file");
		cur++;
	}
	uint32_t num = std::stoul(s);
	if (num == 0) throw std::runtime_error("Incorrect format of file");
	return num;
}

pgm_image::pgm_image(std::string const& filename) {
	std::ifstream input(filename, std::ios_base::binary);
	if (input.fail()) throw std::runtime_error("Could not open input file");
    char input_str[128];
    input.get(input_str, 128, '\n');
	if (strcmp(input_str, "P5") != 0) throw std::runtime_error("Incorrect format of file, expected P5");
	input.ignore();
	input.get(input_str, 128, ' ');
	w = get_number(input_str);
	input.ignore();
	input.get(input_str, 128, '\n');
	h = get_number(input_str);
	input.ignore();
	input.get(input_str, 128, '\n');
	depth = get_number(input_str);
	size_t length = static_cast<size_t>(w) * h;
	try {
		data = std::unique_ptr<uint8_t[]>(new uint8_t[length]);
	} catch (...) {
		throw std::runtime_error("Could not allocate memory");
	}
	input.ignore();
	input.read(reinterpret_cast<char*>(data.get()), length);
	if (input.fail()) throw std::runtime_error("Incorrect format of file");
	input.ignore();
	if (!input.eof()) throw std::runtime_error("Incorrect format of file");
}

void pgm_image::get_histogram(uint32_t(&buf) [256]) {
	for (size_t i = 0; i < 256; i++) buf[i] = 0;
	for (size_t i = 0; i < static_cast<size_t>(w) * h; i++) {
		buf[data[i]]++;
	}
}

void pgm_image::get_histogram(uint32_t(&buf)[256], size_t num_threads) {
	for (size_t i = 0; i < 256; i++) buf[i] = 0;
	if (num_threads != 0) omp_set_num_threads(num_threads);
#pragma omp parallel
	{
		uint32_t cur_buf[256];
		for (size_t j = 0; j < 256; j++) cur_buf[j] = 0;
#pragma omp for
		for (int i = 0; i < static_cast<int>(w) * h; i++) {
			cur_buf[data[i]]++;
		}
#pragma omp critical
		{
			for (size_t j = 0; j < 256; j++) buf[j] += cur_buf[j];
		}
	}
}
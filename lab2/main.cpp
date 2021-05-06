#include <iostream>
#include <fstream>
#include <exception>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

#include "opencl_utils.h"

static uint64_t get_valid_uint_number(char const* s) {
	if (s[0] == '-') throw std::runtime_error("");
	size_t index;
	uint64_t ans = std::stoull(s, &index);
	if (s[index] != '\0') throw std::runtime_error("");
	return ans;
}

static float get_valid_float(char const* s) {
	size_t index;
	float ans = std::stof(s, &index);
	if (s[index] != '\0') throw std::runtime_error("");
	return ans;
}

static void read_file(size_t& n, size_t& k, size_t& m, std::vector<float>& first_matrix, std::vector<float>& second_matrix, char const* input_file) {
	std::ifstream input(input_file);
	if (input.fail()) throw std::runtime_error("Could not open input file");
	try {
		{
			std::string input_str;
			std::getline(input, input_str);
			std::string num;
			size_t index = 0;
			while (index < input_str.length() && input_str[index] != ' ') num += input_str[index++];
			if (index == input_str.length()) throw std::runtime_error("");
			n = get_valid_uint_number(num.c_str());
			num = "";
			index++;
			while (index < input_str.length() && input_str[index] != ' ') num += input_str[index++];
			if (index == input_str.length()) throw std::runtime_error("");
			k = get_valid_uint_number(num.c_str());
			num = "";
			index++;
			while (index < input_str.length()) num += input_str[index++];
			m = get_valid_uint_number(num.c_str());
		}
		if (n == 0 || k == 0 || m == 0) throw std::runtime_error("");
		for (size_t i = 0; i < m; i++) {
			std::string input_str;
			std::getline(input, input_str);
			std::string num;
			size_t index = 0;
			for (size_t j = 0; j < k - 1; j++) {
				while (index < input_str.length() && input_str[index] != ' ') num += input_str[index++];
				if (index == input_str.length()) throw std::runtime_error("");
				first_matrix.push_back(get_valid_float(num.c_str()));
				num = "";
				index++;
			}
			while (index < input_str.length()) num += input_str[index++];
			first_matrix.push_back(get_valid_float(num.c_str()));
		}
		for (size_t i = 0; i < k; i++) {
			std::string input_str;
			std::getline(input, input_str);
			std::string num;
			size_t index = 0;
			for (size_t j = 0; j < n - 1; j++) {
				while (index < input_str.length() && input_str[index] != ' ') num += input_str[index++];
				if (index == input_str.length()) throw std::runtime_error("");
				second_matrix.push_back(get_valid_float(num.c_str()));
				num = "";
				index++;
			}
			while (index < input_str.length()) num += input_str[index++];
			second_matrix.push_back(get_valid_float(num.c_str()));
		}
		std::string input_str;
		std::getline(input, input_str);
		if (input_str != "") throw std::runtime_error("");
	} catch (...) {
		throw std::runtime_error("Incorrect input file");
	}
}

int main(int argc, char* argv[]) {
	if (argc != 5) {
		std::cerr << "Input format: <device's number> <input file> <output file> <implementation's number>" << std::endl;
		return 1;
	}
	uint32_t device_num;
	try {
		device_num = get_valid_uint_number(argv[1]);
	} catch (...) {
		std::cerr << "device's number should be the not negative integer" << std::endl;
		return 1;
	}
	if (argv[4][1] != '\0' || argv[4][0] < '1' || argv[4][0] > '3') {
		std::cerr << "implementation's number should be from 1 to 3" << std::endl;
		return 1;
	}
	size_t n, k, m;
	std::vector<float> first_matrix;
	std::vector<float> second_matrix;
	try {
		read_file(n, k, m, first_matrix, second_matrix, argv[2]);
		std::vector<float> traspose_first_matrix(m * k);
		for (size_t i = 0; i < m; i++) {
			for (size_t j = 0; j < k; j++) {
				traspose_first_matrix[j * m + i] = first_matrix[i * k + j];
			}
		}
		first_matrix = traspose_first_matrix;
		multiply(device_num, argv[4][0], argv[3], n, k, m, first_matrix, second_matrix);
	} catch (std::exception &e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}
	return 0;
}
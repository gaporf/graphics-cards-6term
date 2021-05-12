#include "opencl_utils.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

#include <CL/opencl.h>

size_t const LOCAL_SIZE = 16;
size_t const VECTOR_SIZE = 2;

void multiply(uint32_t device_num, char implement_num, char const* output_file, size_t const& N, size_t const& K, size_t const& M, std::vector<float>& first_matrix, std::vector<float>& second_matrix) {
	auto t_start = std::chrono::high_resolution_clock::now();
	unsigned long long NN = (N % LOCAL_SIZE == 0 ? N : (N / LOCAL_SIZE + 1) * LOCAL_SIZE);
	unsigned long long MM = (M % LOCAL_SIZE == 0 ? M : (M / LOCAL_SIZE + 1) * LOCAL_SIZE);
	unsigned long long KK = (K % LOCAL_SIZE == 0 ? K : (K / LOCAL_SIZE + 1) * LOCAL_SIZE);
	if (implement_num == '3') {
		NN = (NN % (LOCAL_SIZE * VECTOR_SIZE) == 0 ? NN : (NN / (LOCAL_SIZE * VECTOR_SIZE) + 1) * (LOCAL_SIZE * VECTOR_SIZE));
	}
	std::vector<float> new_first_matrix(KK * MM);
	for (size_t i = 0; i < KK; i++) {
		for (size_t j = 0; j < MM; j++) {
			if (i < K && j < M) {
				new_first_matrix[i * MM + j] = first_matrix[i * M + j];
			} else {
				new_first_matrix[i * MM + j] = 0;
			}
		}
	}
	std::vector<float> new_second_matrix(KK * NN);
	for (size_t i = 0; i < KK; i++) {
		for (size_t j = 0; j < NN; j++) {
			if (i < K && j < N) {
				new_second_matrix[i * NN + j] = second_matrix[i * N + j];
			} else {
				new_second_matrix[i * NN + j] = 0;
			}
		}
	}
	cl_uint n;
	cl_int error;
	cl_device_id device = nullptr;
	error = clGetPlatformIDs(0, nullptr, &n);
	if (error != CL_SUCCESS) throw std::runtime_error("Could not get platform ids");
	std::unique_ptr<cl_platform_id[]> platforms;
	try {
		platforms = std::unique_ptr<cl_platform_id[]>(new cl_platform_id[n]);
	} catch (...) {
		throw std::runtime_error("Could not allocate memory");
	}
	error = clGetPlatformIDs(n, platforms.get(), nullptr);
	if (error != CL_SUCCESS) throw std::runtime_error("Could not get platfrom ids");
	uint32_t num_of_devices = 0;
	for (auto i = 0; i < n; i++) {
		cl_uint m;
		error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &m);
		if (error == CL_DEVICE_NOT_FOUND) {
			m = 0;
		} else if (error != CL_SUCCESS) {
			throw std::runtime_error("Could not get device ids");
		}
		num_of_devices += m;
		error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 0, nullptr, &m);
		if (error == CL_DEVICE_NOT_FOUND) {
			m = 0;
		} else if (error != CL_SUCCESS) {
			throw std::runtime_error("Could not get device ids");
		}
		num_of_devices += m;
	}
	if (device_num >= num_of_devices) device_num = 0;
	for (auto i = 0; i < n; i++) {
		cl_uint m;
		error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &m);
		if (error == CL_DEVICE_NOT_FOUND) continue;
		if (error != CL_SUCCESS) throw std::runtime_error("Could not get device ids");
		std::unique_ptr<cl_device_id[]> devices;
		try {
			devices = std::unique_ptr<cl_device_id[]>(new cl_device_id[m]);
		} catch (...) {
			throw std::runtime_error("Could not allocate memory");
		}
		error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, m, devices.get(), nullptr);
		if (error != CL_SUCCESS) throw std::runtime_error("Could not get device ids");
		for (auto j = 0; j < m; j++) {
			size_t k;
			clGetDeviceInfo(devices[j], CL_DEVICE_HOST_UNIFIED_MEMORY, 0, nullptr, &k);
			std::unique_ptr<char[]> type;
			try {
				type = std::unique_ptr<char[]>(new char[k]);
			} catch (...) {
				throw std::runtime_error("Could not allocate memory");
			}
			clGetDeviceInfo(devices[j], CL_DEVICE_HOST_UNIFIED_MEMORY, k, type.get(), nullptr);
			if (*reinterpret_cast<cl_bool*>(type.get()) == CL_FALSE) {
				if (device_num == 0) {
					device = devices[j];
					break;
				} else {
					device_num--;
				}
			}
		}
		if (device != nullptr) break;
	}
	if (device == nullptr) {
		for (auto i = 0; i < n; i++) {
			cl_uint m;
			error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &m);
			if (error == CL_DEVICE_NOT_FOUND) continue;
			if (error != CL_SUCCESS) throw std::runtime_error("Could not get device ids");
			std::unique_ptr<cl_device_id[]> devices;
			try {
				devices = std::unique_ptr<cl_device_id[]>(new cl_device_id[m]);
			} catch (...) {
				throw std::runtime_error("Could not allocate memory");
			}
			error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, m, devices.get(), nullptr);
			if (error != CL_SUCCESS) throw std::runtime_error("Could not get device ids");
			for (auto j = 0; j < m; j++) {
				size_t k;
				clGetDeviceInfo(devices[j], CL_DEVICE_HOST_UNIFIED_MEMORY, 0, nullptr, &k);
				std::unique_ptr<char[]> type;
				try {
					type = std::unique_ptr<char[]>(new char[k]);
				} catch (...) {
					throw std::runtime_error("Could not allocate memory");
				}
				clGetDeviceInfo(devices[j], CL_DEVICE_HOST_UNIFIED_MEMORY, k, type.get(), nullptr);
				if (*reinterpret_cast<cl_bool*>(type.get()) == CL_TRUE) {
					if (device_num == 0) {
						device = devices[j];
						break;
					} else {
						device_num--;
					}
				}
			}
			if (device != nullptr) break;
		}
	}
	if (device == nullptr) {
		for (auto i = 0; i < n; i++) {
			cl_uint m;
			error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 0, nullptr, &m);
			if (error == CL_DEVICE_NOT_FOUND) continue;
			if (error != CL_SUCCESS) throw std::runtime_error("Could not get device ids");
			std::unique_ptr<cl_device_id[]> devices;
			try {
				devices = std::unique_ptr<cl_device_id[]>(new cl_device_id[m]);
			} catch (...) {
				throw std::runtime_error("Could not allocate memory");
			}
			error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, m, devices.get(), nullptr);
			if (error != CL_SUCCESS) throw std::runtime_error("Could not get device ids");
			for (auto j = 0; j < m; j++) {
				if (device_num == 0) {
					device = devices[j];
					break;
				} else {
					device_num--;
				}
			}
			if (device != nullptr) break;
		}
	}
	if (device == nullptr) throw std::runtime_error("Could not find device");
	{
		size_t m;
		error = clGetDeviceInfo(device, CL_DEVICE_VENDOR, 0, nullptr, &m);
		if (error != CL_SUCCESS) throw std::runtime_error("Could not get device vendor");
		std::unique_ptr<char[]> name;
		try {
			name = std::unique_ptr<char[]>(new char[m]);
		} catch (...) {
			throw std::runtime_error("Could not allocate memory");
		}
		error = clGetDeviceInfo(device, CL_DEVICE_VENDOR, m, name.get(), nullptr);
		if (error != CL_SUCCESS) throw std::runtime_error("Could not get device vendor");
		std::cout << reinterpret_cast<char*>(name.get()) << std::endl;
		error = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &m);
		if (error != CL_SUCCESS) throw std::runtime_error("Could not get device name");
		try {
			name = std::unique_ptr<char[]>(new char[m]);
		} catch (...) {
			throw std::runtime_error("Could not allocate memory");
		}
		error = clGetDeviceInfo(device, CL_DEVICE_NAME, m, name.get(), nullptr);
		if (error != CL_SUCCESS) throw std::runtime_error("Could not get device name");
		std::cout << reinterpret_cast<char*>(name.get()) << std::endl;
	}
	cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
	if (error != CL_SUCCESS) throw std::runtime_error("Could not create context");
	cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
	if (error != CL_SUCCESS) throw std::runtime_error("Could not create command queue");
	std::string code_program;
	if (implement_num == '1') {
		code_program += "kernel void sum(global const float *at, global const float *b, global float *c, ulong const N, ulong const K, ulong const M) {\n";
		code_program += "    ulong x = get_global_id(0);\n";
		code_program += "    ulong y = get_global_id(1);\n";
		code_program += "    c[y * N + x] = 0;\n";
		code_program += "    for (uint i = 0; i < K; i++) {\n";
		code_program += "        c[y * N + x] += at[i * M + y] * b[i * N + x];\n";
		code_program += "    }\n";
		code_program += "}\n";
	} else if (implement_num == '2') {
		code_program += "kernel void sum(global const float *at, global const float *b, global float *c, ulong const N, ulong const K, ulong const M) {\n";
		code_program += "    local float first_arr[" + std::to_string(LOCAL_SIZE) + "][" + std::to_string(LOCAL_SIZE) + "];\n";
		code_program += "    local float second_arr[" + std::to_string(LOCAL_SIZE) + "][" + std::to_string(LOCAL_SIZE) + "];\n";
		code_program += "    ulong x = get_global_id(0);\n";
		code_program += "    ulong x_local = get_local_id(0);\n";
		code_program += "    ulong y = get_global_id(1);\n";
		code_program += "    ulong y_local = get_local_id(1);\n";
		code_program += "    c[y * N + x] = 0;\n";
		code_program += "    for (ulong i = 0; i < K; i += " + std::to_string(LOCAL_SIZE) + ") {\n";
		code_program += "        first_arr[y_local][x_local] = at[(i + x_local) * M + y];\n";
		code_program += "        second_arr[y_local][x_local] = b[(i + y_local) * N + x];\n";
		code_program += "        barrier(CLK_LOCAL_MEM_FENCE);\n";
		code_program += "        for (ulong uu = 0; uu < " + std::to_string(LOCAL_SIZE) + "; uu++) {\n";
		code_program += "            c[y * N + x] += first_arr[y_local][uu] * second_arr[uu][x_local];\n";
		code_program += "        }\n";
		code_program += "    }\n";
		code_program += "}\n";
	} else {
		code_program += "kernel void sum(global const float *at, global const float *b, global float *c, ulong const N, ulong const K, ulong const M) {\n";
		code_program += "    local float first_arr[" + std::to_string(LOCAL_SIZE) + "][" + std::to_string(LOCAL_SIZE) + "];\n";
		code_program += "    local float" + std::to_string(VECTOR_SIZE) + " second_arr[" + std::to_string(LOCAL_SIZE) + "][" + std::to_string(LOCAL_SIZE) + "];\n";
		code_program += "    ulong x = get_global_id(0);\n";
		code_program += "    ulong x_local = get_local_id(0);\n";
		code_program += "    ulong y = get_global_id(1);\n";
		code_program += "    ulong y_local = get_local_id(1);\n";
		code_program += "    float" + std::to_string(VECTOR_SIZE) + " ans = 0;\n";
		code_program += "    for (ulong i = 0; i < K; i += " + std::to_string(LOCAL_SIZE) + ") {\n";
		code_program += "        first_arr[y_local][x_local] = at[(i + x_local) * M + y];\n";
		code_program += "        second_arr[y_local][x_local] = vload" + std::to_string(VECTOR_SIZE) + "(0, &b[(i + y_local) * N + " + std::to_string(VECTOR_SIZE) + " * x]);\n";
		code_program += "        barrier(CLK_LOCAL_MEM_FENCE);\n";
		code_program += "        for (ulong uu = 0; uu < " + std::to_string(LOCAL_SIZE) + "; uu++) {\n";
		code_program += "            ans += first_arr[y_local][uu] * second_arr[uu][x_local];\n";
		code_program += "        }\n";
		code_program += "    }\n";
		code_program += "    vstore" + std::to_string(VECTOR_SIZE) + "(ans, 0, &c[y * N + " + std::to_string(VECTOR_SIZE) + " * x]);\n";
		code_program += "}\n";
	}
	char const* code = code_program.c_str();
	cl_program program = clCreateProgramWithSource(context, 1, &code, nullptr, &error);
	if (error != CL_SUCCESS) throw std::runtime_error("Could not create program");
	error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
	if (error != CL_SUCCESS) {
		size_t q;
		error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &q);
		if (error != CL_SUCCESS) throw std::runtime_error("Could not get build info");
		std::unique_ptr<char[]> log;
		try {
			log = std::unique_ptr<char[]>(new char[q]);
		} catch (...) {
			throw std::runtime_error("Could not allocate memory");
		}
		error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, q, log.get(), nullptr);
		if (error != CL_SUCCESS) throw std::runtime_error("Could not get build info (2)");
		std::cout << log.get() << std::endl;
		throw std::runtime_error("Could not build program");
	}
	cl_kernel kernel = clCreateKernel(program, "sum", &error);
	if (error != CL_SUCCESS) throw std::runtime_error("Could not create kernel");
	cl_mem buffer_a = clCreateBuffer(context, CL_MEM_READ_WRITE, MM * KK * 4, nullptr, &error);
	if (error != CL_SUCCESS) throw std::runtime_error("Could not create buffer a");
	error = clEnqueueWriteBuffer(queue, buffer_a, CL_FALSE, 0, MM * KK * 4, new_first_matrix.data(), 0, nullptr, nullptr);
	if (error != CL_SUCCESS) throw std::runtime_error("Could not write buffer a");
	cl_mem buffer_b = clCreateBuffer(context, CL_MEM_READ_WRITE, KK * NN * 4, nullptr, &error);
	if (error != CL_SUCCESS) throw std::runtime_error("Could not create buffer b");
	error = clEnqueueWriteBuffer(queue, buffer_b, CL_FALSE, 0, KK * NN * 4, new_second_matrix.data(), 0, nullptr, nullptr);
	if (error != CL_SUCCESS) throw std::runtime_error("Could not write buffer b");
	cl_mem buffer_c = clCreateBuffer(context, CL_MEM_READ_WRITE, MM * NN * 4, nullptr, &error);
	if (error != CL_SUCCESS) throw std::runtime_error("Could not create buffer c");
	error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_a);
	if (error != CL_SUCCESS) throw std::runtime_error("Could not set first argument");
	error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_b);
	if (error != CL_SUCCESS) throw std::runtime_error("Could not set second argument");
	error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_c);
	if (error != CL_SUCCESS) throw std::runtime_error("Could not set third argument");
	error = clSetKernelArg(kernel, 3, sizeof(unsigned long long), &NN);
	if (error != CL_SUCCESS) throw std::runtime_error("Could not set fourth argument");
	error = clSetKernelArg(kernel, 4, sizeof(unsigned long long), &KK);
	if (error != CL_SUCCESS) throw std::runtime_error("Could not set fifth argument");
	error = clSetKernelArg(kernel, 5, sizeof(unsigned long long), &MM);
	if (error != CL_SUCCESS) throw std::runtime_error("Could not set sixth argument");
	size_t const dim[2] = { (implement_num == '3' ? NN / VECTOR_SIZE : NN), MM };
	size_t const local_size[2] = { LOCAL_SIZE, LOCAL_SIZE };
	size_t const* local_ptr = (implement_num == '1' ? nullptr : local_size);
	cl_event event;
	error = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, dim, local_ptr, 0, nullptr, &event);
	if (error != CL_SUCCESS) throw std::runtime_error("Could not enqueue ND range kernel");
	std::vector<float> cl_ans(MM * NN);
	error = clEnqueueReadBuffer(queue, buffer_c, CL_TRUE, 0, MM * NN * 4, cl_ans.data(), 0, nullptr, nullptr);
	if (error != CL_SUCCESS) throw std::runtime_error("Could not enque read buffer");
	auto t_end = std::chrono::high_resolution_clock::now();
	std::ofstream output(output_file);
	if (output.fail()) throw std::runtime_error("Could not open file for writing");
	output << N << " " << M << std::endl;
	for (size_t i = 0; i < M; i++) {
		for (size_t j = 0; j < N; j++) {
			output << cl_ans[i * NN + j];
			if (j + 1 != N) {
				output << " ";
			} else {
				output << std::endl;
			}
		}
	}
	cl_ulong start_time, finish_time;
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, nullptr);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &finish_time, nullptr);
	std::cout << "\nTime: " << static_cast<double>(finish_time - start_time) / 1'000'000 << "\t" << std::chrono::duration<double, std::milli>(t_end - t_start).count() << "\n";
}
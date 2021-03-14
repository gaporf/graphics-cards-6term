#ifndef PGM_IMAGE_H
#define PGM_IMAGE_H

#include <memory>
#include <cstdint>
#include <string>
#include <array>

struct pgm_image {
	pgm_image(std::string const& filename);

	pgm_image(pgm_image const&) = delete;

	pgm_image& operator=(pgm_image const&) = delete;

	~pgm_image() = default;

	void get_histogram(uint32_t (&buf)[256]);

	void get_histogram(uint32_t(&buf)[256], size_t num_threads);

private:
	std::unique_ptr<uint8_t[]> data;
	uint32_t w;
	uint32_t h;
	uint32_t depth;
};

#endif
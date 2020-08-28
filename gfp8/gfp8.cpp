/*
Copyright 2020, Yves Gallot

gfp8 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.

gfp8 searches for Generalized Fermat Progressions with length >= 8: numbers b such that b^{2^k} + 1 are primes for k = 0...7.
The integer sequence is https://oeis.org/A337364.
*/

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <inttypes.h>

#include <xmmintrin.h>
#include <x86intrin.h>

#include <gmp.h>
#include <omp.h>

// #define VALID	true
// #define PROFILE	true
// #define PROFILE_COUNT	true

// #define CHECK_COUNT	5

// #define AVX_512

#if defined (_WIN32)	// use Performance Counter
#include <Windows.h>
#else					// otherwise use gettimeofday() instead
#include <sys/time.h>
#endif

struct timer
{
#if defined (_WIN32)
	typedef LARGE_INTEGER time;
#else
	typedef timeval time;
#endif

	static time currentTime()
	{
#if defined (_WIN32)
		LARGE_INTEGER time; QueryPerformanceCounter(&time);
#else
		timeval time; gettimeofday(&time, nullptr);
#endif
		return time;
	}

	static double diffTime(const time & end, const time & start)
	{
#if defined (_WIN32)
		LARGE_INTEGER freq; QueryPerformanceFrequency(&freq);
		return double(end.QuadPart - start.QuadPart) / double(freq.QuadPart);
#else
		return double(end.tv_sec - start.tv_sec) + double(end.tv_usec - start.tv_usec) * 1e-6;
#endif
	}
};

static std::string header()
{
	const char * const sysver =
#if defined(_WIN64)
		"win64";
#elif defined(_WIN32)
		"win32";
#elif defined(__linux__)
#ifdef __x86_64
		"linux64";
#else
		"linux32";
#endif
#elif defined(__APPLE__)
		"macOS";
#else
		"unknown";
#endif

	std::ostringstream ssc;
#if defined(__GNUC__)
	ssc << " gcc-" << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#elif defined(__clang__)
	ssc << " clang-" << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#endif

	std::ostringstream ss;
	ss << "gfp8 0.9.5 " << sysver << ssc.str() << std::endl;
	ss << "Copyright (c) 2020, Yves Gallot" << std::endl;
	ss << "gfp8 is free source code, under the MIT license." << std::endl;
	ss << std::endl;
	return ss.str();
}

static std::string usage()
{
	std::ostringstream ss;
	ss << "Usage: gfp8 [b_min] [b_max] [numThreads]" << std::endl;
	ss << "         b_min is the start of the b search (in P (10^15) values, default 0)" << std::endl;
	ss << "         b_max is the end of the b search (in P (10^15) values, default 2^95)" << std::endl;
	ss << "         numThreads is the number of threads (default 0 = one thread per logical CPU)" << std::endl << std::endl;
	return ss.str();
}

static void to_string(const __uint128_t n, char * const str)
{
	if (n > UINT64_MAX)
	{
		const uint64_t P10_UINT64 = 10000000000000000000ull;
		const uint64_t leading = uint64_t(n / P10_UINT64), trailing = uint64_t(n % P10_UINT64);
		sprintf(str, "%" PRIu64 "%.19" PRIu64, leading, trailing);
	}
	else
	{
		sprintf(str, "%" PRIu64, uint64_t(n));
	}
}

inline uint32_t mulmod32(const uint32_t x, const uint32_t y, const uint32_t m)
{
	return uint32_t((x * uint64_t(y)) % m);
}

inline __uint128_t mul_hi_95(const __uint128_t x, const uint64_t y_l, const uint32_t y_h)
{
	const uint64_t x_l = uint64_t(x); const uint32_t x_h = uint32_t(x >> 64);
	const __uint128_t z_l = x_l * __uint128_t(y_l);
	const uint64_t z_h = x_h * uint64_t(y_h);
	const __uint128_t z_m = ((z_l >> 64) | (__uint128_t(z_h) << 64)) + y_h * __uint128_t(x_l) + x_h * __uint128_t(y_l);
	return z_m >> 31;
}

#include "mods_p.h"

static void sieveP(uint8_t * const sieve, const size_t size, const uint32_t p)
{
	for (size_t i = 0; i < size; ++i)
	{
		uint32_t r = uint32_t(i % p);
		if (r == p - 1) sieve[i] = 1;
		for (int n = 1; n <= 7; ++n)
		{
			r = mulmod32(r, r, p);
			if (r == p - 1) sieve[i] = 1;
		}
	}
}

static void sieveP1(uint8_t * const sieve, const uint32_t p)
{
	for (size_t i = 0; i < size_t(p); ++i) sieve[i] = 0;
	sieveP(sieve, size_t(p), p);
}

static void sieveP2(uint8_t * const sieve, const uint32_t p1, const uint32_t p2)
{
	const size_t p1p2 = p1 * size_t(p2);
	for (size_t i = 0; i < p1p2; ++i) sieve[i] = 0;
	sieveP(sieve, p1p2, p1); sieveP(sieve, p1p2, p2);
}

#define def_sieve(P)		static uint8_t sieve_##P[P]; sieveP1(sieve_##P, P)
#define check_sieve(b, P)	if (sieve_##P[mod_##P(b)] != 0) continue;

#define check_sv(P, ind)	(sieve_##P[b_p[ind]] != 0)

inline void mpz_set_ui_128(mpz_t rop, const __uint128_t n)
{
	const uint64_t n_h = uint64_t(n >> 64);
	const size_t count = (n_h == 0) ? 1 : 2;
	mp_limb_t * const p_limb = mpz_limbs_write(rop, count);
	p_limb[0] = uint64_t(n);
	if (n_h != 0) p_limb[1] = n_h;
	mpz_limbs_finish(rop, count);
}

#ifdef VALID
#include "valid.h"
static bool is_valid = true;
#endif

static void output(const __uint128_t b, const int n, const std::string & extension)
{
	char b_str[64]; to_string(b, b_str);

	static std::mutex output_mutex;
	const std::lock_guard<std::mutex> lock(output_mutex);

	if (b == 1) std::cout << b_str << std::endl;
	else
	{
#ifndef VALID
		std::cout << "                                               \r";
#endif
		std::cout << b_str << "\t" << "GFP-" << n << std::endl;

		const std::string logFilename = "gfp8" + extension + std::string(".log");
		std::ofstream logFile(logFilename, std::ios::app);
		if (logFile.is_open())
		{
			logFile << b_str << "\t" << "GFP-" << n << std::endl;
			logFile.flush();
			logFile.close();
		}
	}

	if (n >= 8)
	{
		const std::string resFilename = "gfp8" + extension + std::string(".res");
		std::ofstream resFile(resFilename, std::ios::app);
		if (resFile.is_open())
		{
			resFile << b_str << std::endl;
			resFile.flush();
			resFile.close();
		}
	}
}

int main(int argc, char * argv[])
{
	std::cout << header();
	std::cout << usage();

	const long long b_min = (argc > 1) ? std::atoll(argv[1]) : 0;
	const long long b_max = (argc > 2) ? std::atoll(argv[2]) : 39614081257133ll;
	const int numThreads = (argc > 3) ? std::atoi(argv[3]) : 0;

	std::stringstream ss; ss << "_" << b_min << "_" << b_max;
	const std::string extension = ss.str();

	// weights: 257: 0.0077821, 17: 0.117647, 5: 0.4, 2: 0.5, 3: 0.666667
	// 6426, 15420, 17476, 21846, 32640, 32896, 43690, 48060, 50116, 59110, 65536, 76330, 91750, 104856, 120276, 131070 = 2 * 3 * 5 * 17 * 257
	static const size_t pattern_size = 16;
	static const size_t pattern_mod = 131070;
	static const uint16_t pattern_step[pattern_size] = { 6426, 8994, 2056, 4370, 10794, 256, 10794, 4370, 2056, 8994, 6426, 10794, 15420, 13106, 15420, 10794 };

#ifdef AVX_512
	const size_t vsize = 32;
	typedef uint16_t vec[vsize] __attribute__((aligned(64)));							// zmm registers
	static const vec step_p = { 7 * 97, 13 * 41, 769, 193, 641, 11 * 37, 23 * 29, 449, 113, 1153, 577, 73, 1409, 353, 19 * 31, 89,
								3329, 241, 53, 1217, 137, 61, 2689, 673, 337, 1601, 401, 3457, 433, 929, 7681, 7937 };
#else
	const size_t vsize = 16;
	typedef uint16_t vec[vsize] __attribute__((aligned(32)));							// xmm or ymm registers
	static const vec step_p = { 7 * 97, 13 * 41, 769, 193, 641, 11 * 37, 23 * 29, 449, 113, 1153, 577, 73, 1409, 353, 19 * 31, 89 };
#endif
	static vec pattern_step_p[pattern_size];
	for (size_t j = 0; j < pattern_size; ++j)
	{
		vec & psj = pattern_step_p[j];
		for (size_t i = 0; i < vsize; ++i) psj[i] = pattern_step[j] % step_p[i];
	}

	static uint8_t sieve_7_97[7 * 97], sieve_13_41[13 * 41], sieve_11_37[11 * 37], sieve_23_29[23 * 29], sieve_19_31[19 * 31];
	sieveP2(sieve_7_97, 7, 97); sieveP2(sieve_13_41, 13, 41); sieveP2(sieve_11_37, 11, 37); sieveP2(sieve_23_29, 23, 29); sieveP2(sieve_19_31, 19, 31);
	
#include "def_sieves.hc"

	mpz_t two; mpz_init_set_ui(two, 2);

	__uint128_t b_start = __uint128_t(1e15 * b_min), b_end = __uint128_t(1e15 * b_max) + 1;

	if (numThreads != 0)
	{
		omp_set_dynamic(0);
		omp_set_num_threads(numThreads);
	}

	size_t n_thread = 1;
#if !defined(PROFILE) && !defined(VALID)
#pragma omp parallel
{
	n_thread = omp_get_num_threads();
}
	std::cout << n_thread << " thread(s)" << std::endl;
#endif

	const size_t slice = size_t(1) << 24;

	__uint128_t b_ctx = 0;
	{
		const std::string ctxFilename = "gfp8" + extension + std::string(".ctx");
		std::ifstream ctxFile(ctxFilename);
		if (ctxFile.is_open())
		{
			uint64_t b_ctx_l = 0, b_ctx_h = 0;
			ctxFile >> b_ctx_l; ctxFile >> b_ctx_h;
			b_ctx = b_ctx_l | (__uint128_t(b_ctx_h) << 64);
			ctxFile.close();
		}
	}

	const bool resume = (b_start < b_ctx);
	if (resume) b_start = b_ctx;
	b_start /= pattern_mod; b_start *= pattern_mod;
	b_end /= pattern_mod; b_end *= pattern_mod; b_end += slice * pattern_mod * n_thread;
	if ((b_end >> 95) != 0) b_end = __uint128_t(1) << 95;

	char b_start_str[64]; to_string(b_start, b_start_str);
	char b_end_str[64]; to_string(b_end, b_end_str);
	std::cout << (resume ? "Resuming from a checkpoint, t" : "T") << "esting from " << b_start_str << " to " << b_end_str << std::endl;

	if (b_start == 0) output(1, 8, extension);

	const timer::time start_time = timer::currentTime();
#ifdef PROFILE
	uint64_t dt_min = INT64_MAX, dt_max = 0;
#endif

	for (__uint128_t b_g = b_start; b_g < b_end; b_g += slice * pattern_mod * n_thread)
	{
		{
			const double dt = timer::diffTime(timer::currentTime(), start_time);
			if (dt > 1)
			{
				char b_g_str[64]; to_string(b_g, b_g_str);
				std::cout << b_g_str << ", " << int((b_g - b_start) * 86400.0 * 1e-15 / dt) << " P/day             \r" << std::flush;
			}

			const std::string ctxFilename = "gfp8" + extension + std::string(".ctx");
			std::ofstream ctxFile(ctxFilename);
			if (ctxFile.is_open())
			{
				const uint64_t b_g_l = uint64_t(b_g), b_g_h = uint64_t(b_g >> 64);
				ctxFile << b_g_l << " " << b_g_h << std::endl;
				ctxFile.flush();
				ctxFile.close();
			}
		}

#if !defined(PROFILE) && !defined(VALID)
#pragma omp parallel for
#endif
		for (size_t j = 0; j < n_thread; ++j)
		{
			__uint128_t b = b_g + j * slice * pattern_mod;
			vec b_p; for (size_t k = 0; k < vsize; ++k) b_p[k] = uint16_t(b % step_p[k]);

			mpz_t b2n, r; mpz_inits(b2n, r, nullptr);

#ifdef PROFILE
#ifdef PROFILE_COUNT
			size_t count[4]; for (size_t i = 0; i < 4; ++i) count[i] = 0;
#endif
			_mm_lfence();
			const uint64_t t0 = __rdtsc();
#endif
			for (size_t i = 0, i_pattern = 0; i < slice * pattern_size; ++i, i_pattern = (i_pattern + 1) % pattern_size)
			{
				const vec & psi = pattern_step_p[i_pattern];

#ifdef AVX_512
#pragma omp simd aligned(b_p, psi, step_p : 64) simdlen(32)		// generates AVX-512 instructions
#else
#pragma omp simd aligned(b_p, psi, step_p : 32)					// generates SSE2 or AVX2 instructions
#endif
				for (size_t k = 0; k < vsize; ++k)
				{
					const uint16_t r = b_p[k] + psi[k];
					const uint16_t p = step_p[k];
					b_p[k] = (r >= p) ? r - p : r;
				}

				b += pattern_step[i_pattern];
#ifdef VALID
				if (!is_valid) { std::cout << "error" << std::endl; exit(1); }
				is_valid = false;
				if ((j != 0) || (i >= test_size)) break;
				b = b_test[i];
				for (size_t i = 0; i < vsize; ++i) b_p[i] = uint16_t(b % step_p[i]);
				bool bpat = false;
				uint64_t pattern_val = 0;
				for (size_t j = 0; j < pattern_size; ++j)
				{
					if (b % pattern_mod == pattern_val) bpat = true;
					pattern_val += pattern_step[j];
				}
				if (!bpat) { std::cout << "pattern error" << std::endl; exit(1); }
				std::cout << i + 1 << ": ";
#endif
#if CHECK_COUNT == 4
				if (check_sv(7_97, 0) | check_sv(13_41, 1) | check_sv(769, 2) | check_sv(193, 3)) continue;
				if (check_sv(641, 4) | check_sv(11_37, 5) | check_sv(23_29, 6) | check_sv(449, 7)) continue;
				if (check_sv(113, 8) | check_sv(1153, 9) | check_sv(577, 10) | check_sv(73, 11)) continue;
				if (check_sv(1409, 12) | check_sv(353, 13) | check_sv(19_31, 14) | check_sv(89, 15)) continue;
#elif CHECK_COUNT == 6
				if (check_sv(7_97, 0) | check_sv(13_41, 1) | check_sv(769, 2) | check_sv(193, 3) | check_sv(641, 4) | check_sv(11_37, 5)) continue;
				if (check_sv(23_29, 6) | check_sv(449, 7) | check_sv(113, 8) | check_sv(1153, 9) | check_sv(577, 10)) continue;
				if (check_sv(73, 11) | check_sv(1409, 12) | check_sv(353, 13) | check_sv(19_31, 14) | check_sv(89, 15)) continue;
#elif CHECK_COUNT == 8
				if (check_sv(7_97, 0) | check_sv(13_41, 1) | check_sv(769, 2) | check_sv(193, 3)
				  | check_sv(641, 4) | check_sv(11_37, 5) | check_sv(23_29, 6) | check_sv(449, 7)) continue;
				if (check_sv(113, 8) | check_sv(1153, 9) | check_sv(577, 10) | check_sv(73, 11)
				  | check_sv(1409, 12) | check_sv(353, 13) | check_sv(19_31, 14) | check_sv(89, 15)) continue;
#else // CHECK_COUNT == 5
				if (check_sv(7_97, 0) | check_sv(13_41, 1) | check_sv(769, 2) | check_sv(193, 3) | check_sv(641, 4)) continue;

				// 13.5 cycles, 13.4%
#ifdef PROFILE_COUNT
				count[0] += 1;
#endif
				if (check_sv(11_37, 5) | check_sv(23_29, 6) | check_sv(449, 7) | check_sv(113, 8) | check_sv(1153, 9)) continue;

				// 15.5 cycles, 6.4%
#ifdef PROFILE_COUNT
				count[1] += 1;
#endif
				if (check_sv(577, 10) | check_sv(73, 11) | check_sv(1409, 12) | check_sv(353, 13) | check_sv(19_31, 14) | check_sv(89, 15)) continue;

				// 16.5 cycles, 3.6%
#ifdef PROFILE_COUNT
				count[2] += 1;
#endif
#endif

#ifdef AVX_512
				if (check_sv(3329, 16 + 0) | check_sv(241, 16 + 1) | check_sv(53, 16 + 2) | check_sv(1217, 16 + 3) | check_sv(137, 16 + 4)) continue;
				if (check_sv(61, 16 + 5) | check_sv(2689, 16 + 6) | check_sv(673, 16 + 7) | check_sv(337, 16 + 8) | check_sv(1601, 16 + 9)) continue;
				if (check_sv(401, 16 + 10) | check_sv(3457, 16 + 11) | check_sv(433, 16 + 12) | check_sv(929, 16 + 13) | check_sv(7681, 16 + 14) | check_sv(7937, 16 + 15)) continue;

#include "check_sieves_512.hc"
#else
#include "check_sieves.hc"
#endif

				// 28.5 cycles, 0.2%
#ifdef PROFILE_COUNT
				count[3] += 1;
#endif
				if ((b & (b - 1)) == 0) continue;	// power of two are 2-prp
				mpz_set_ui_128(b2n, b);

				const int n_min = 6;
				int n = 0;
				while (n < n_min)
				{
					mpz_add_ui(r, b2n, 1);
					mpz_powm(r, two, b2n, r);
					if (mpz_cmp_ui(r, 1) != 0) break;
					mpz_mul(b2n, b2n, b2n);
					++n;
				}

				if (n == n_min)
				{
					mpz_set_ui_128(b2n, b);

					n = 0;
					while (true)
					{
						mpz_add_ui(r, b2n, 1);
						if (mpz_probab_prime_p(r, 10) == 0) break;
						mpz_mul(b2n, b2n, b2n);
						++n;
					}

					if (n >= n_min)
					{
						output(b, n, extension);
#ifdef VALID
						is_valid = true;
#endif
					}
				}

				// 37.5 cycles
			}

#ifdef PROFILE
			const uint64_t dt = __rdtsc() - t0;
			dt_min = std::min(dt_min, dt); dt_max = std::max(dt_max, dt);
			std::cout << dt / double(slice * pattern_size) << " cycles, min = " << dt_min << ", max = " << dt_max;
#ifdef PROFILE_COUNT
			for (size_t i = 0; i < 4; ++i) std::cout << ", " << 100.0 * count[i] / double(slice * pattern_size) << "%";
#endif
			std::cout << std::endl;
#endif
			mpz_clears(b2n, r, nullptr);
		}
#ifdef VALID
		break;
#endif
	}

	mpz_clear(two);

	return EXIT_SUCCESS;
}

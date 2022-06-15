/*
Copyright 2022, Yves Gallot

xgfp8 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.

xgfp8 searches for extended Generalized Fermat Progressions with length >= 8: numbers a, b such that a^{2^k} + b^{2^k} are primes for k = 0...7.
The integer sequence is https://oeis.org/A343121.
*/

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <mutex>

#include <vector>
#include <deque>
#include <functional>

#include <gmp.h>

#include "prm.h"

#define	XGFP	8
// #define	GEN_SIEVE

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
	ss << "xgfp8 0.1.0 " << sysver << ssc.str() << std::endl;
	ss << "Copyright (c) 2022, Yves Gallot" << std::endl;
	ss << "xgfp8 is free source code, under the MIT license." << std::endl;
	ss << std::endl;
	return ss.str();
}

static std::string usage()
{
	std::ostringstream ss;
	ss << "Usage: xgfp8 <a_min> <a_max>" << std::endl;
	ss << "   search in [a_min; a_max] (default a_min = 2, a_max = 2^32 - 1)" << std::endl << std::endl;
	return ss.str();
}

inline uint32_t addmod32(const uint32_t x, const uint32_t y, const uint32_t m)
{
	const uint32_t c = (x >= m - y) ? m : 0;
	return x + y - c;
}

static void fill_sieve(bool * const sieve, const uint32_t p)
{
	for (size_t i = 0; i <  p * size_t(p); ++i) sieve[i] = 0;

	for (uint32_t a = 0; a < p; ++a)
	{
		for (uint32_t b = 0; b <= a; ++b)
		{
			if (addmod32(a, b, p) == 0)
			{
				sieve[a * size_t(p) + b] = sieve[b * size_t(p) + a] = 1;
				continue;
			}
			uint32_t ra = a, rb = b;
			for (size_t n = 1; n <= XGFP - 1; ++n)
			{
				ra = (ra * ra) % p; rb = (rb * rb) % p;
				if (addmod32(ra, rb, p) == 0)
				{
					sieve[a * size_t(p) + b] = sieve[b * size_t(p) + a] = 1;
					break;
				}
			}
		}
	}
}

#define def_sieve(P)			bool * sieve_##P = new bool[P * size_t(P)]; fill_sieve(sieve_##P, P)
#define check_sieve(a, b, P)	if (sieve_##P[(a % P) * P + (b % P)] != 0) continue

static void output(const uint32_t a, const uint32_t b, const int n)
{
	static std::mutex output_mutex;
	const std::lock_guard<std::mutex> lock(output_mutex);

	std::cout << "                               \r" << a << ", " << b << "\t" << "xGFP-" << n << std::endl;

	if (n >= XGFP)
	{
		std::ofstream resFile("xgfp8.log", std::ios::app);
		if (resFile.is_open())
		{
			resFile << a << ", " << b << std::endl;
			resFile.flush();
			resFile.close();
		}
	}
}

#ifdef GEN_SIEVE
inline int ilog2(const uint64_t x) { return 63 - __builtin_clzll(x); }

inline uint64_t addmod64(const uint64_t x, const uint64_t y, const uint64_t m)
{
	const uint64_t c = (x >= m - y) ? m : 0;
	return x + y - c;
}

inline uint64_t mul_hi(const uint64_t x, const uint64_t y) { return uint64_t((x * __uint128_t(y)) >> 64); }

// Barrett's product: let n = 63, r = ceil(log2(p)), p_shift = r - 2 = ceil(log2(p)) - 1, t = n + 1 = 64,
// p_inv = floor(2^(s + t) / p). Then the number of iterations h = 1.
// We must have x^2 < alpha.p with alpha = 2^(n-2). If p <= 2^(n-2) = 2^61 then x^2 < p^2 <= alpha.p.

inline uint64_t barrett_inv(const uint64_t p, int & p_shift)
{
	p_shift = ilog2(p) - 1;
	return uint64_t((__uint128_t(1) << (p_shift + 64)) / p);
}

inline uint64_t barrett_mul(const uint64_t x, const uint64_t y, const uint64_t p, const uint64_t p_inv, const int p_shift)
{
	const __uint128_t xy = x * __uint128_t(y);
	uint64_t r = uint64_t(xy) - mul_hi(uint64_t(xy >> p_shift), p_inv) * p;
	if (r >= p) r -= p;
	return r;
}

inline bool prp(const uint64_t n)	// n must be odd, 2-prp test
{
	const uint64_t e = n - 1;

	int s; const uint64_t q = barrett_inv(n, s);

	int b = ilog2(e) - 1;
	uint64_t mask = uint64_t(1) << b;

	uint64_t r = 2;
	if ((n >> 32) != 0)
	{
		r *= r;							// r = 2^2
		if ((e & mask) != 0) r += r;	// r <= 2^3
		--b; mask >>= 1;

		r *= r;							// r <= 2^6
		if ((e & mask) != 0) r += r;	// r <= 2^7
		--b; mask >>= 1;

		r *= r;							// r <= 2^14
		if ((e & mask) != 0) r += r;	// r <= 2^15
		--b; mask >>= 1;

		r *= r;							// r <= 2^30
		if ((e & mask) != 0) r += r;	// r <= 2^31
		--b; mask >>= 1;
	}

	while (b >= 0)
	{
		r = barrett_mul(r, r, n, q, s);
		if ((e & mask) != 0) r = addmod64(r, r, n);
		--b; mask >>= 1;
	}

	return (r == 1);
}

static double sieve_weight(const bool * const sieve, const uint32_t p)
{
	const size_t size = p * size_t(p);
	size_t weight = 0;
	for (size_t i = 0; i < size; ++i) weight += (sieve[i] == 0);
	// std::cout << p << ", " << weight << ", " << size << ", " << weight / double(size) << std::endl;
	return weight / double(size);
}

static void gen_sieve(const size_t count)
{
	std::vector<std::pair<uint32_t, double>> weights;

	const uint32_t p_max = 1500;
	bool * const sieve = new bool[p_max * size_t(p_max)];

	for (uint32_t p = 3; p <= p_max; p += 2)
	{
		if (prp(p))
		{
			fill_sieve(sieve, p);
			const double w = sieve_weight(sieve, p);
			weights.push_back(std::make_pair(p, w));
			// std::cout << p << ": " << w << std::endl;
		}
	}

	delete[] sieve;

	std::sort(weights.begin(), weights.end(), [&](const auto & p1, const auto & p2) { return p1.second < p2.second; });

	// for (size_t i = 0; i < count; ++i) std::cout << weights[i].first << ": " << weights[i].second << std::endl;
	for (size_t i = 0; i < count; ++i) std::cout << "def_sieve(" << weights[i].first << ");" << std::endl;
	for (size_t i = 0; i < count; ++i) std::cout << "check_sieve(a, b, " << weights[i].first << ");" << std::endl;
}
#endif

int main(int argc, char * argv[])
{
	std::cout << header();
	std::cout << usage();

	const uint32_t a_min = (argc > 1) ? uint32_t(std::atoll(argv[1])) : 2;
	const uint32_t a_max = (argc > 2) ? uint32_t(std::atoll(argv[2])) : uint32_t(-1);

#ifdef GEN_SIEVE
	gen_sieve(100);
	return EXIT_SUCCESS;
#endif

	def_sieve(257); def_sieve(17); def_sieve(5); def_sieve(3);
	def_sieve(769); def_sieve(193); def_sieve(97); def_sieve(13);
	def_sieve(641); def_sieve(41); def_sieve(7); def_sieve(449);
	def_sieve(113); def_sieve(1153); def_sieve(577); def_sieve(29);
	def_sieve(73); def_sieve(11); def_sieve(1409); def_sieve(353);
	def_sieve(37); def_sieve(89); def_sieve(241); def_sieve(53);
	def_sieve(19); def_sieve(1217); def_sieve(137); def_sieve(61);
	def_sieve(23); def_sieve(31); def_sieve(43); def_sieve(47);
	def_sieve(59); def_sieve(67); def_sieve(71); def_sieve(79);

	uint32_t a_ctx = 0;
	std::ifstream ctxFile("xgfp8.ctx");
	if (ctxFile.is_open())
	{
		ctxFile >> a_ctx;
		ctxFile.close();
	}

	uint32_t a_start = a_min, a_end = a_max;
	const bool resume = (a_start < a_ctx);
	if (resume) a_start = a_ctx;

	std::cout << (resume ? "Resuming from a checkpoint, t" : "T") << "esting from " << a_start << " to " << a_end << std::endl;

	std::deque<uint32_t> primes;
	PrmGen prmgen; uint32_t prm =  prmgen.first();
	for (; prm <= 2 * a_start; prm = prmgen.next())
	{
		if (prm >= a_start) primes.push_back(prm);	// a <= p = a + b <= 2a
	}

	mpz_t two, a2n, b2n, xgfn, r; mpz_inits(two, a2n, b2n, xgfn, r, nullptr);
	mpz_set_ui(two, 2);

	timer::time disp_time = timer::currentTime();
	size_t count = 0;

	for (uint32_t a = a_start; a <= a_end; ++a)
	{
		const timer::time cur_time = timer::currentTime();
		const double dt = timer::diffTime(cur_time, disp_time);
		if (dt > 10)
		{
			disp_time = cur_time;
			std::cout << a << ", +" << int(count * 86400.0 / 1e0 / dt) << "/day       \r" << std::flush;
			count = 0;

			std::ofstream ctxFile("xgfp8.ctx");
			if (ctxFile.is_open())
			{
				ctxFile << a << std::endl;
				ctxFile.flush();
				ctxFile.close();
			}
		}
		++count;

		for (const uint32_t p : primes)
		{
			const uint32_t b = p - a;

			check_sieve(a, b, 257); check_sieve(a, b, 17); check_sieve(a, b, 5); check_sieve(a, b, 3);
			check_sieve(a, b, 769); check_sieve(a, b, 193); check_sieve(a, b, 97); check_sieve(a, b, 13);
			check_sieve(a, b, 641); check_sieve(a, b, 41); check_sieve(a, b, 7); check_sieve(a, b, 449);
			check_sieve(a, b, 113); check_sieve(a, b, 1153); check_sieve(a, b, 577); check_sieve(a, b, 29);
			check_sieve(a, b, 73); check_sieve(a, b, 11); check_sieve(a, b, 1409); check_sieve(a, b, 353);
			check_sieve(a, b, 37); check_sieve(a, b, 89); check_sieve(a, b, 241); check_sieve(a, b, 53);
			check_sieve(a, b, 19); check_sieve(a, b, 1217); check_sieve(a, b, 137); check_sieve(a, b, 61);
			check_sieve(a, b, 23); check_sieve(a, b, 31); check_sieve(a, b, 43); check_sieve(a, b, 47);
			check_sieve(a, b, 59); check_sieve(a, b, 67); check_sieve(a, b, 71); check_sieve(a, b, 79);

			mpz_set_ui(a2n, a); mpz_set_ui(b2n, b);

			const int n_min = XGFP - 1;
			int n = 1;
			mpz_mul(a2n, a2n, a2n);
			mpz_mul(b2n, b2n, b2n);
			while (n < n_min)
			{
				mpz_add(xgfn, a2n, b2n);
				mpz_sub_ui(r, xgfn, 1);
				mpz_powm(r, two, r, xgfn);
				if (mpz_cmp_ui(r, 1) != 0) break;
				mpz_mul(a2n, a2n, a2n);
				mpz_mul(b2n, b2n, b2n);
				++n;
			}

			if (n == n_min)
			{
				mpz_set_ui(a2n, a); mpz_set_ui(b2n, b);

				n = 0;
				while (true)
				{
					mpz_add(xgfn, a2n, b2n);
					if (mpz_probab_prime_p(xgfn, 10) == 0) break;
					mpz_mul(a2n, a2n, a2n);
					mpz_mul(b2n, b2n, b2n);
					++n;
				}

				if (n >= n_min) output(a, b, n);
			}
		}

		// a <= p <= 2a => a + 1 <= p <= 2a + 2
		if (primes.front() == a) primes.pop_front();
		if (prm == 2 * a + 1)
		{
			primes.push_back(prm);
		 	prm = prmgen.next();
		}
	}

	mpz_clears(two, a2n, b2n, xgfn, r, nullptr);

	return EXIT_SUCCESS;
}

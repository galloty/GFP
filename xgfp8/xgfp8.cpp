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

inline uint32_t gcd(const uint32_t x, const uint32_t y)
{
	uint32_t a = x, b = y;
	while (b != 0)
	{
		const size_t t = b;
		b = a % b;
		a = t;
	}
	return a;
}

static size_t fill_sieve(bool * const sieve, const uint32_t m)
{
	size_t count = 0;

	for (size_t i = 0; i < m * size_t(m); ++i) sieve[i] = false;

	for (uint32_t a = 0; a < m; ++a)
	{
		for (uint32_t b = 0; b <= a; ++b)
		{
			const uint32_t ab = addmod32(a, b, m);
			if ((ab == 0) || (gcd(ab, m) != 1))
			{
				sieve[a * size_t(m) + b] = sieve[b * size_t(m) + a] = true;
				count += (a == b) ? 1 : 2;
				continue;
			}
			uint32_t ra = a, rb = b;
			for (size_t n = 1; n <= XGFP - 1; ++n)
			{
				ra = (ra * ra) % m; rb = (rb * rb) % m;
				const uint32_t rab = addmod32(ra, rb, m);
				if ((rab == 0) || (gcd(rab, m) != 1))
				{
					sieve[a * size_t(m) + b] = sieve[b * size_t(m) + a] = true;
					count += (a == b) ? 1 : 2;
					break;
				}
			}
		}
	}

	// for (size_t i = 0; i < m; ++i)
	// {
	// 	for (size_t j = 0; j < m; ++j)
	// 	{
	// 		std::cout << (sieve[i * size_t(m) + j] ? 1 : 0) << " ";
	// 	}
	// 	std::cout << std::endl;
	// }

	return m * size_t(m) - count;
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

static void gen_sieve()
{
	std::vector<std::pair<uint32_t, double>> weights;

	const uint32_t p_max = 1500;
	bool * const sieve = new bool[p_max * size_t(p_max)];
	PrmGen prmgen; prmgen.first();
	for (uint32_t p = prmgen.next(); p <= p_max; p = prmgen.next())
	{
		const double weight = fill_sieve(sieve, p) / (p * double(p));
		weights.push_back(std::make_pair(p, weight));
		std::cout << p << ": " << weight << std::endl;
	}
	delete[] sieve;

	std::sort(weights.begin(), weights.end(), [&](const auto & p1, const auto & p2) { return p1.second < p2.second; });

	const size_t count = 100;
	for (size_t i = 0; i < count; ++i) std::cout << weights[i].first << ": " << weights[i].second << std::endl;
	// for (size_t i = 0; i < count; ++i) std::cout << "def_sieve(" << weights[i].first << ");" << std::endl;
	// for (size_t i = 0; i < count; ++i) std::cout << "check_sieve(a, b, " << weights[i].first << ");" << std::endl;
}
#endif

static double next_a(const double a, const double dcount)
{
	const double count = 0.5 * a * a / log(a) + dcount;

	double a_min = a, a_max = 1e12;
	while (fabs(a_max - a_min) >= 1)
	{
		const double a_half = 0.5 * (a_min + a_max);
		if (0.5 * a_half * a_half / log(a_half) < count) a_min = a_half; else a_max = a_half;
	}

	return 0.5 * (a_min + a_max);
}

int main(int argc, char * argv[])
{
	std::cout << header();
	std::cout << usage();

	const uint32_t a_min = (argc > 1) ? uint32_t(std::atoll(argv[1])) : 2;
	const uint32_t a_max = (argc > 2) ? uint32_t(std::atoll(argv[2])) : uint32_t(-1);

#ifdef GEN_SIEVE
	gen_sieve();
	return EXIT_SUCCESS;
#endif

	// 255 = 3 * 5 * 17, 679 = 7 * 97, 533 = 13 * 41, 407 = 11 * 37, 667 = 23 * 29, 589 = 19 * 31
	def_sieve(257); def_sieve(255); def_sieve(679); def_sieve(533);
	def_sieve(769); def_sieve(193); def_sieve(641); def_sieve(407);
	def_sieve(667); def_sieve(449); def_sieve(113); def_sieve(1153);
	def_sieve(577);	def_sieve(73); def_sieve(1409); def_sieve(353);
	def_sieve(589);	def_sieve(89); def_sieve(241); def_sieve(53);
	def_sieve(1217); def_sieve(137); def_sieve(61); def_sieve(673);

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

	std::vector<uint32_t> primes;
	PrmGen prmgen; uint32_t prm =  prmgen.first();
	for (; prm <= 2 * a_start; prm = prmgen.next())
	{
		if (prm >= a_start) primes.push_back(prm);	// a <= p = a + b <= 2a
	}

	mpz_t two, a2n, b2n, xgfn, r; mpz_inits(two, a2n, b2n, xgfn, r, nullptr);
	mpz_set_ui(two, 2);

	timer::time disp_time = timer::currentTime();
	uint32_t disp_a = a_start;
	size_t pcount = 0, scount = 0;

	for (uint32_t a = a_start; a <= a_end; ++a)
	{
		const timer::time cur_time = timer::currentTime();
		const double dt = timer::diffTime(cur_time, disp_time);
		if (dt > 10)
		{
			const double count = 0.5 * (a / log(a) * a - disp_a / log(disp_a) * disp_a);
			const double na = next_a(a, count * 86400.0 / dt);
			std::cout << a << ", +" << std::setprecision(3) << 1e-6 * (na - a) << "M/day, prime ratio: 1/"
				<< a / primes.size() << ", sieve ratio: 1/" << pcount / scount << "       \r" << std::flush;
			disp_time = cur_time; disp_a = a;

			std::ofstream ctxFile("xgfp8.ctx");
			if (ctxFile.is_open())
			{
				ctxFile << a << std::endl;
				ctxFile.flush();
				ctxFile.close();
			}
		}

		// std:: cout << "hello";
		const uint32_t * const p = primes.data();
		for (size_t i = 0, s = primes.size(); i < s; ++i)
		{
			const uint32_t b = p[i] - a;
			++pcount;

			check_sieve(a, b, 257); check_sieve(a, b, 255); check_sieve(a, b, 679); check_sieve(a, b, 533);
			check_sieve(a, b, 769); check_sieve(a, b, 193); check_sieve(a, b, 641); check_sieve(a, b, 407);
			check_sieve(a, b, 667); check_sieve(a, b, 449); check_sieve(a, b, 113); check_sieve(a, b, 1153);
			check_sieve(a, b, 577);	check_sieve(a, b, 73); check_sieve(a, b, 1409); check_sieve(a, b, 353);
			check_sieve(a, b, 589);	check_sieve(a, b, 89); check_sieve(a, b, 241); check_sieve(a, b, 53);
			check_sieve(a, b, 1217); check_sieve(a, b, 137); check_sieve(a, b, 61); check_sieve(a, b, 673);

			++scount;

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
		if (primes.front() == a) primes.erase(primes.begin());
		if (prm == 2 * a + 1)
		{
			primes.push_back(prm);
		 	prm = prmgen.next();
		}
	}

	mpz_clears(two, a2n, b2n, xgfn, r, nullptr);

	return EXIT_SUCCESS;
}

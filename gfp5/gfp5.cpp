/*
Copyright 2020, Yves Gallot

gfp5 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.

gfp5 searches for Generalized Fermat Progressions with length >= 5: numbers b such that b^{2^k} + 1 are primes for k = 0...4.
The integer sequence is https://oeis.org/A070694.
*/

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>

#include <gmp.h>

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
	ss << "gfp5 1.0.0 " << sysver << ssc.str() << std::endl;
	ss << "Copyright (c) 2020, Yves Gallot" << std::endl;
	ss << "gfp5 is free source code, under the MIT license." << std::endl;
	ss << std::endl;
	return ss.str();
}

static std::string usage()
{
	std::ostringstream ss;
	ss << "Usage: gfp5 [b_min]    b_min is the start of the b search (in T (10^12) values, default 0)" << std::endl << std::endl;
	return ss.str();
}

inline uint32_t mulmod32(const uint32_t x, const uint32_t y, const uint32_t m)
{
	return uint32_t((x * uint64_t(y)) % m);
}

inline uint64_t dupmod64(const uint64_t x, const uint64_t m)
{
	uint64_t r = x + x;
	if (x >= m - x) r -= m;
	return r;
}

inline int ilog2(const uint64_t x) { return 63 - __builtin_clzll(x); }

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
		if ((e & mask) != 0) r = dupmod64(r, n);
		--b; mask >>= 1;
	}

	return (r == 1);
}

static void sieveP(uint8_t * const sieve, const uint32_t p)
{
	for (uint32_t i = 0; i < p - 1; ++i)
	{
		sieve[i] = 0;
		uint32_t r = i;
		for (size_t n = 1; n <= 4; ++n)
		{
			r = mulmod32(r, r, p);
			if (r == p - 1) sieve[i] = 1;
		}
	}
	sieve[p - 1] = 1;
}

#define def_sieve(P)		static uint8_t sieve_##P[P]; sieveP(sieve_##P, P)
#define check_sieve(b, P)	sieve_##P[b % P] == 1

static void output(const uint64_t b, const int n)
{
	if (b == 1) std::cout << b << std::endl;
	else std::cout << b << "\t" << "GFP-" << n << std::endl;

	std::ofstream resFile("gfp5.log", std::ios::app);
	if (resFile.is_open())
	{
		resFile << b << std::endl;
		resFile.flush();
		resFile.close();
	}

	std::ofstream ctxFile("gfp5.ctx");
	if (ctxFile.is_open())
	{
		ctxFile << b << std::endl;
		ctxFile.flush();
		ctxFile.close();
	}
}

int main(int argc, char * argv[])
{
	std::cout << header();

	std::cout << usage();

	const int b_min = (argc > 1) ? std::atoi(argv[1]) : 0;

	// weights: 17: 0.117647, 5: 0.4, 2: 0.5, 3: 0.666667
	// The pattern is 0, 120, 136, 256, 306, 340, 426, 460 (mod 510 = 2 * 3 * 5 * 17)
	static const size_t pattern_size = 8;
	static const uint16_t pattern_mod = 510;
	static const uint16_t pattern_step[pattern_size] = { 120, 16, 120, 50, 34, 86, 34, 50 };

	// weights: 97: 0.680412, 13: 0.769231, 41: 0.829268
	def_sieve(97); def_sieve(13); def_sieve(41);
	// weights: 193: 0.839378, 7: 0.857143, 113: 0.867257
	def_sieve(193); def_sieve(7); def_sieve(113);
	// weights: 257: 0.879377, 29: 0.896552, 73: 0.90411
	def_sieve(257); def_sieve(29); def_sieve(73);
	// weights: 11: 0.909091, 353: 0.912181, 37: 0.918919
	def_sieve(11); def_sieve(353); def_sieve(37);
	// weights: 89: 0.921348, 449: 0.930958, 241: 0.937759
	def_sieve(89); def_sieve(449); def_sieve(241);
	// weights: 53: 0.943396, 577: 0.946274, 19: 0.947368
	def_sieve(53); def_sieve(577); def_sieve(19);
	// weights: 137: 0.948905, 61: 0.95082, 641: 0.951638
	def_sieve(137); def_sieve(61); def_sieve(641);
	// weights: 673: 0.953938, 337: 0.95549, 23: 0.956522
	def_sieve(673); def_sieve(337); def_sieve(23);
	// weights: 769: 0.959688, 401: 0.962594, 433: 0.965358
	def_sieve(769); def_sieve(401); def_sieve(433);
	// weights: 929: 0.966631, 31: 0.967742, 233: 0.969957
	def_sieve(929); def_sieve(31); def_sieve(233);
	// weights: 101: 0.970297, 109: 0.972477, 1153: 0.973114
	def_sieve(101); def_sieve(109); def_sieve(1153);
	// weights: 1217: 0.974528, 593: 0.974705, 281: 0.975089
	def_sieve(1217); def_sieve(593); def_sieve(281);
	// weights: 1249: 0.97518, 43: 0.976744, 313: 0.977636
	def_sieve(1249); def_sieve(43); def_sieve(313);
	// weights: 1409: 0.977999, 47: 0.978723, 149: 0.979866
	def_sieve(1409); def_sieve(47); def_sieve(149);

	mpz_t b2n, gfn, two, r; mpz_inits(b2n, gfn, two, r, nullptr);
	mpz_set_ui(two, 2);

	uint64_t b_start = uint64_t(1e12 * b_min); b_start /= pattern_mod; b_start *= pattern_mod;

	uint64_t b_ctx = 0;
	std::ifstream ctxFile("gfp5.ctx");
	if (ctxFile.is_open())
	{
		ctxFile >> b_ctx;
		ctxFile.close();
	}

	const bool resume = (b_start < b_ctx);
	if (resume) { b_start = b_ctx; b_start /= pattern_mod; b_start *= pattern_mod; }

	std::cout << (resume ? "Resuming from a checkpoint, t" : "T") << "esting from " << b_start << std::endl;

	uint64_t b = b_start;

	if (b == 0)
	{
		output(1, 5);
		output(2, 5);
	}

	for (size_t i = 0; true; ++i)
	{
		b += pattern_step[i % pattern_size];
		if (b >= (1ull << 61)) break;

		if (check_sieve(b, 97)) continue;
		if (check_sieve(b, 13)) continue;
		if (check_sieve(b, 41)) continue;

		if (check_sieve(b, 193)) continue;
		if (check_sieve(b, 7)) continue;
		if (check_sieve(b, 113)) continue;

		if (check_sieve(b, 257)) continue;
		if (check_sieve(b, 29)) continue;
		if (check_sieve(b, 73)) continue;

		if (check_sieve(b, 11)) continue;
		if (check_sieve(b, 353)) continue;
		if (check_sieve(b, 37)) continue;

		if (check_sieve(b, 89)) continue;
		if (check_sieve(b, 449)) continue;
		if (check_sieve(b, 241)) continue;

		if (check_sieve(b, 53)) continue;
		if (check_sieve(b, 577)) continue;
		if (check_sieve(b, 19)) continue;

		if (check_sieve(b, 137)) continue;
		if (check_sieve(b, 61)) continue;
		if (check_sieve(b, 641)) continue;

		if (check_sieve(b, 673)) continue;
		if (check_sieve(b, 337)) continue;
		if (check_sieve(b, 23)) continue;

		if (check_sieve(b, 769)) continue;
		if (check_sieve(b, 401)) continue;
		if (check_sieve(b, 433)) continue;

		if (check_sieve(b, 929)) continue;
		if (check_sieve(b, 31)) continue;
		if (check_sieve(b, 233)) continue;

		if (check_sieve(b, 101)) continue;
		if (check_sieve(b, 109)) continue;
		if (check_sieve(b, 1153)) continue;

		if (check_sieve(b, 1217)) continue;
		if (check_sieve(b, 593)) continue;
		if (check_sieve(b, 281)) continue;

		if (check_sieve(b, 1249)) continue;
		if (check_sieve(b, 43)) continue;
		if (check_sieve(b, 313)) continue;

		if (check_sieve(b, 1409)) continue;
		if (check_sieve(b, 47)) continue;
		if (check_sieve(b, 149)) continue;

		if ((b & (b - 1)) == 0) continue;	// power of two
		if (!prp(b + 1)) continue;

		mpz_set_ui(b2n, 1); b2n->_mp_d[0] = b;

		int n = -1;
		do
		{
			++n;
			if (n == 4) break;
			mpz_mul(b2n, b2n, b2n);
			mpz_add_ui(gfn, b2n, 1);
			mpz_powm(r, two, b2n, gfn);
		} while (mpz_cmp_ui(r, 1) == 0);	// 2-prp

		if (n >= 4)
		{
			mpz_set_ui(b2n, 1); b2n->_mp_d[0] = b;

			n = -1;
			while (true)
			{
				mpz_add_ui(r, b2n, 1);
				if (mpz_probab_prime_p(r, 10) == 0) break;
				mpz_mul(b2n, b2n, b2n);
				++n;
			}

			if (n >= 4) output(b, n + 1);
		}
	}

	mpz_clears(b2n, gfn, two, r, nullptr);

	return EXIT_SUCCESS;
}

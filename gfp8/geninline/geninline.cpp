/*
Copyright 2020, Yves Gallot

gfp8 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <list>

#include "prmgen.h"

inline int ilog2(const uint64_t x) { return 63 - __builtin_clzll(x); }

// Torbjörn Granlund and Peter L. Montgomery, Division by Invariant Integers Using Multiplication
// In Proceedings of the SIGPLAN '94 Conference on Programming Language Design and Implementation, 1994, p 61-72

struct GM95 { __uint128_t m; int sh; };

static GM95 choose_multiplier_95(const uint64_t d)	// d is not a power of 2
{
	const int l = ilog2(d) + 1;
	const __uint128_t m = ((__uint128_t(1) << (95 + l)) + (__uint128_t(1) << l)) / d;
	if ((m >> 96) != 0) { std::cout << "Error:" << d << std::endl; exit(1); }
	GM95 gm; gm.m = m; gm.sh = l;
	return gm;
}

inline __uint128_t mul_hi_95(const __uint128_t x, const uint64_t y_l, const uint32_t y_h)
{
	const uint64_t x_l = uint64_t(x); const uint32_t x_h = uint32_t(x >> 64);
	const __uint128_t z_l = x_l * __uint128_t(y_l);
	const uint64_t z_h = x_h * uint64_t(y_h);
	const __uint128_t z_m = ((z_l >> 64) | (__uint128_t(z_h) << 64)) + y_h * __uint128_t(x_l) + x_h * __uint128_t(y_l);
	return z_m >> 31;
}

// static __uint128_t gm_div_95(const __uint128_t n, const GM95 & d_inv)
// {
// 	return mul_hi_95(n, uint64_t(d_inv.m), uint32_t(d_inv.m >> 64)) >> d_inv.sh;
// }

static void gm_print_95(const int p, std::ofstream & s)
{
	const GM95 p_inv = choose_multiplier_95(p);
	s << "inline size_t mod_" << p << "(const __uint128_t n) { ";
	s << "const __uint128_t d = mul_hi_95(n, " << uint64_t(p_inv.m) << "ull, " << uint32_t(p_inv.m >> 64) << "u) >> ";
	s << p_inv.sh << "; return n - d * " << p << "; }" << std::endl;
}

static void gen_primes(std::vector<uint32_t> & primes, const size_t count)
{
	PrmGen prmGen;

	std::list<std::pair<uint32_t, double>> weight;

	const int n = 8;

	for (uint32_t p = uint32_t(prmGen.first()); p < 1000000; p = uint32_t(prmGen.next()))
	{
		if ((p == 2) || (p == 3) || (p == 5) || (p == 17) || (p == 257)) continue;		// Fermat primes
		//  7 * 97: 0.857143 * 0.680412 = 0.583210
		// 13 * 41: 0.769231 * 0.829268 = 0.637899
		// 11 * 37: 0.909091 * 0.918919 = 0.835381
		// 23 * 29: 0.956522 * 0.896552 = 0.857572
		// 19 * 31: 0.947368 * 0.967742 = 0.916808
		if ((p <= 41) || (p == 97)) continue;

		uint32_t w_p;
		if (p == 2) w_p = 1;
		else
		{
			int e = 0;
			for (uint32_t k = p - 1; k % 2 == 0; k /= 2) ++e;
			e = std::min(e, n);
			w_p = (uint32_t(1) << e) - 1;
		}

		const double W_p = (1.0 - w_p / double(p));
		weight.push_back(std::make_pair(p, W_p));
	}

	weight.sort([](const std::pair<uint32_t, double> & a, const std::pair<uint32_t, double> & b) { return a.second < b.second; } );

	primes.clear();

	size_t size = 0, c = 0;
	uint32_t p_max = 0;
	for (const auto & pair : weight)
	{
		const uint32_t p = pair.first;
		primes.push_back(p);
		size += p;
		p_max = std::max(p_max, p);

		int e = 0;
		uint32_t k = p - 1;
		while ((e < n) && (k % 2 == 0)) { ++e; k /= 2; }
		const uint32_t w_p = (p == 2) ? 1 : (uint32_t(1) << e) - 1;
		std::cout << p << " = " << k << "*2^" << e << " + 1, w_p = " << w_p << ", 1 - w_p / p = " << pair.second << std::endl;
		if (++c == count) break;
	}

	std::cout << "size = " << size << ", p_max = " << p_max << std::endl;
}

int main()
{
	const size_t count = 250;
	std::vector<uint32_t> primes;

	gen_primes(primes, count);

	std::ofstream dsFile("../def_sieves.hc");
	if (dsFile.is_open())
	{
		for (const uint32_t & p : primes)
		{
			dsFile << "def_sieve(" << p << ");" << std::endl;
		}
		dsFile.close();
	}

	std::ofstream csFile("../check_sieves.hc");
	if (csFile.is_open())
	{
		for (const uint32_t & p : primes)
		{
			// 769, 193, 641, 449, 113, 1153, 577, 29, 73, 11, 1409, 353, 37, 89 are included in the fast sieve
			if ((p == 769) || (p == 193) || (p == 641) || (p == 449) || (p == 113) || (p == 1153)) continue;
			if ((p == 577) || (p == 29) || (p == 73) || (p == 11) || (p == 1409) || (p == 353) || (p == 37) || (p == 89)) continue;
			csFile << "check_sieve(b, " << p << ");" << std::endl;
		}
		csFile.close();
	}

	std::ofstream mpFile("../mods_p.h");
	if (mpFile.is_open())
	{
		for (const uint32_t & p : primes)
		{
			// 769, 193, 641, 449, 113, 1153, 577, 29, 73, 11, 1409, 353, 37, 89 are included in the fast sieve
			if ((p == 769) || (p == 193) || (p == 641) || (p == 449) || (p == 113) || (p == 1153)) continue;
			if ((p == 577) || (p == 29) || (p == 73) || (p == 11) || (p == 1409) || (p == 353) || (p == 37) || (p == 89)) continue;
			gm_print_95(p, mpFile);
		}
	}

	return EXIT_SUCCESS;
}

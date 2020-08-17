/*
Copyright 2020, Yves Gallot

gfp7 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.

gfp7 searches for Generalized Fermat Progressions with length >= 7: numbers b such that b^{2^k} + 1 are primes for k = 0...6.
The integer sequence is https://oeis.org/A335805.
*/

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <mutex>

#include <gmp.h>
#include <omp.h>

// #define VALID	true

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
	ss << "gfp7 1.0.0 " << sysver << ssc.str() << std::endl;
	ss << "Copyright (c) 2020, Yves Gallot" << std::endl;
	ss << "gfp7 is free source code, under the MIT license." << std::endl;
	ss << std::endl;
	return ss.str();
}

static std::string usage()
{
	std::ostringstream ss;
	ss << "Usage: gfp7 [b_min] [numThreads]" << std::endl;
	ss << "         b_min is the start of the b search (in T (10^12) values, default 0)" << std::endl;
	ss << "         numThreads is the number of threads (default 0 = one thread per logical CPU)" << std::endl << std::endl;
	return ss.str();
}

inline uint32_t mulmod32(const uint32_t x, const uint32_t y, const uint32_t m)
{
	return uint32_t((x * uint64_t(y)) % m);
}

inline uint64_t addmod64(const uint64_t x, const uint64_t y, const uint64_t m)
{
	const uint64_t c = (x >= m - y) ? m : 0;
	return x + y - c;
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
	if ((n >> 32) == 0)
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

static void sieveP(uint8_t * const sieve, const size_t size, const uint32_t p)
{
	for (size_t i = 0; i < size; ++i)
	{
		uint32_t r = uint32_t(i % p);
		if (r == p - 1) sieve[i] = 1;
		for (int n = 1; n <= 6; ++n)
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
#define check_sieve(b, P)	sieve_##P[b % P] == 1

static void output(const uint64_t b, const int n)
{
	static std::mutex output_mutex;
	const std::lock_guard<std::mutex> lock(output_mutex);

	if (b == 1) std::cout << b << std::endl;
	else std::cout << b << "\t" << "GFP-" << n << std::endl;

	if (n >= 7)
	{
		std::ofstream resFile("gfp7.log", std::ios::app);
		if (resFile.is_open())
		{
			resFile << b << std::endl;
			resFile.flush();
			resFile.close();
		}
	}
}

int main(int argc, char * argv[])
{
	std::cout << header();

	std::cout << usage();

	const int b_min = (argc > 1) ? std::atoi(argv[1]) : 0;
	const int numThreads = (argc > 2) ? std::atoi(argv[2]) : 0;

#ifdef VALID
	static const size_t test_size = 55;
	static const uint64_t b_test[test_size] = { 2072005925466ull, 5082584069416ull, 12698082064890ull, 29990491969260ull, 46636691707050ull,
		65081025897426ull, 83689703895606ull, 83953213480290ull, 105003537341346ull, 105699143244090ull, 107581715369910ull, 111370557491826ull,
		111587899569066ull, 128282713771996ull, 133103004825210ull, 154438435790340ull, 161027983349016ull, 169081476836850ull, 199763736975426ull,
		201852155494656ull, 261915314595636ull, 268118920455760ull, 286556471193256ull, 301118628461886ull, 312235790115726ull,	324131153489776ull,
		355306576579540ull, 359868138215170ull, 360451719089070ull, 382897414818370ull, 400403462797566ull, 405113387071246ull, 463603317696946ull,
		473260661898880ull, 476479174179736ull, 482872515565840ull, 487843898732940ull, 529270045932306ull, 529904637878730ull, 533849609566396ull,
		593750085647950ull, 634749607136296ull, 638895561059136ull, 674246937239076ull, 675332667126276ull, 681789036366390ull, 687515678867046ull,
		702469331078776ull, 727781457982660ull, 732245491589670ull, 740290229100376ull, 796774361853936ull, 811756826917290ull, 831001337306046ull,
		240164550712338756ull };
#endif

	// weights: 17: 0.117647, 5: 0.4, 2: 0.5, 3: 0.666667
	// The pattern is 0, 120, 136, 256, 306, 340, 426, 460 (mod 510 = 2 * 3 * 5 * 17)
	static const size_t pattern_size = 8;
	static const size_t pattern_mod = 510;
	static const uint16_t pattern_step[pattern_size] = { 120, 16, 120, 50, 34, 86, 34, 50 };

	// weights: 257: 0.505837, 7 * 97: 0.583210 (97: 0.680412, 7: 0.857143), 13 * 41: 0.6378986 (13: 0.769231, 41: 0.829268)
	static uint8_t sieve_7_97[7 * 97], sieve_13_41[13 * 41];
	def_sieve(257); sieveP2(sieve_7_97, 7, 97); sieveP2(sieve_13_41, 13, 41);
	// weights: 193: 0.673575, 641: 0.801872, 769: 0.83485
	def_sieve(193); def_sieve(641); def_sieve(769);
	// weights: 449: 0.859688, 113: 0.867257, 1153: 0.889853
	def_sieve(449); def_sieve(113); def_sieve(1153);
	// weights: 577: 0.890815, 29: 0.896552, 73: 0.90411
	def_sieve(577); def_sieve(29); def_sieve(73);
	// weights: 11: 0.909091, 1409: 0.909865, 353: 0.912181
	def_sieve(11); def_sieve(1409); def_sieve(353);
	// weights: 37: 0.918919, 89: 0.921348, 241: 0.937759
	def_sieve(37); def_sieve(89); def_sieve(241);
	// weights: 53: 0.943396, 19: 0.947368, 1217: 0.948233
	def_sieve(53); def_sieve(19); def_sieve(1217);
	// weights: 137: 0.948905, 61: 0.95082, 2689: 0.952771
	def_sieve(137); def_sieve(61); def_sieve(2689);
	// weights: 673: 0.953938, 337: 0.95549, 23: 0.956522
	def_sieve(673); def_sieve(337); def_sieve(23);
	// weights: 1601: 0.96065, 3329: 0.96185, 401: 0.962594
	def_sieve(1601); def_sieve(3329); def_sieve(401);
	// weights: 3457: 0.963263, 433: 0.965358, 929: 0.966631
	def_sieve(3457); def_sieve(433); def_sieve(929);
	// weights: 31: 0.967742, 233: 0.969957, 2113: 0.970185
	def_sieve(31); def_sieve(233); def_sieve(2113);
	// weights: 101: 0.970297, 4481: 0.971658, 109: 0.972477
	def_sieve(101); def_sieve(4481); def_sieve(109);
	// weights: 4993: 0.974564, 593: 0.974705, 281: 0.975089
	def_sieve(4993); def_sieve(593); def_sieve(281);
	// weights: 1249: 0.97518, 43: 0.976744, 2753: 0.977116
	def_sieve(1249); def_sieve(43); def_sieve(2753);
	// weights: 313: 0.977636, 47: 0.978723, 149: 0.979866
	def_sieve(313); def_sieve(47); def_sieve(149);
	// weights: 3137: 0.979917, 6529: 0.980548, 157: 0.980892
	def_sieve(3137); def_sieve(6529); def_sieve(157);
	// weights: 1697: 0.981732, 7297: 0.982596, 173: 0.982659
	def_sieve(1697); def_sieve(7297); def_sieve(173);
	// weights: 409: 0.982885, 881: 0.982974, 59: 0.983051
	def_sieve(409); def_sieve(881); def_sieve(59);
	// weights: 181: 0.983425, 7681: 0.983466, 1889: 0.983589
	def_sieve(181); def_sieve(7681); def_sieve(1889);

	mpz_t two; mpz_init_set_ui(two, 2);

	uint64_t b_start = uint64_t(1e12 * b_min); b_start /= pattern_mod; b_start *= pattern_mod;

	if (numThreads != 0)
	{
		omp_set_dynamic(0);
		omp_set_num_threads(numThreads);
	}

	size_t n_thread = 1;
#pragma omp parallel
{
	n_thread = omp_get_num_threads();
}
	std::cout << n_thread << " thread(s)" << std::endl;

	uint64_t b_ctx = 0;
	std::ifstream ctxFile("gfp7.ctx");
	if (ctxFile.is_open())
	{
		ctxFile >> b_ctx;
		ctxFile.close();
	}

	const bool resume = (b_start < b_ctx);
	if (resume) b_start = b_ctx;
	std::cout << (resume ? "Resuming from a checkpoint, t" : "T") << "esting from " << b_start << std::endl;

	if (b_start == 0) output(1, 7);

	const timer::time start_time = timer::currentTime();

	const size_t slice = size_t(1) << 25;

	for (uint64_t b_g = b_start, b_g_step = slice * pattern_mod * n_thread; b_g < (1ull << 61) + b_g_step; b_g += b_g_step)
	{
		const double dt = timer::diffTime(timer::currentTime(), start_time);
		if (dt > 1) std::cout << int((b_g - b_start) * 86400.0 * 1e-12 / dt) << " T/day\r" << std::flush;

		std::ofstream ctxFile("gfp7.ctx");
		if (ctxFile.is_open())
		{
			ctxFile << b_g << std::endl;
			ctxFile.flush();
			ctxFile.close();
		}

#pragma omp parallel for
		for (size_t j = 0; j < n_thread; ++j)
		{
			uint64_t b = b_g + j * slice * pattern_mod;
			size_t b_257 = b % 257, b_7_97 = b % (7 * 97), b_13_41 = b % (13 * 41), b_193 = b % 193;

			mpz_t b2n, r; mpz_init(b2n); mpz_init(r);

			for (size_t i = 0; i < slice * pattern_size; ++i)
			{
				const size_t s = pattern_step[i % pattern_size];

				b += s;
				b_257 = addmod64(b_257, s, 257);	// s < 257
				b_7_97 = addmod64(b_7_97, s, 7 * 97);
				b_13_41 = addmod64(b_13_41, s, 13 * 41);
				b_193 = addmod64(b_193, s, 193);	// s < 193

				if (b >= (1ull << 61)) break;
#ifdef VALID
				if ((j != 0) || (i >= test_size)) break;
				b = b_test[i];
				b_257 = b % 257, b_7_97 = b % (7 * 97), b_13_41 = b % (13 * 41), b_193 = b % 193;
				bool bpat = false;
				uint64_t pattern_val = 0;
				for (size_t j = 0; j < pattern_size; ++j)
				{
					if (b % pattern_mod == pattern_val) bpat = true;
					pattern_val += pattern_step[j];
				}
				if (!bpat)
				{
					std::cout << "pattern error" << std::endl;
					break;
				}
				std::cout << i + 1 << ": ";
#endif
				if ((sieve_257[b_257] == 1) | (sieve_7_97[b_7_97] == 1) | (sieve_13_41[b_13_41] == 1) | (sieve_193[b_193] == 1)) continue;

				if (check_sieve(b, 641)) continue;
				if (check_sieve(b, 769)) continue;

				if (check_sieve(b, 449)) continue;
				if (check_sieve(b, 113)) continue;
				if (check_sieve(b, 1153)) continue;

				if (check_sieve(b, 577)) continue;
				if (check_sieve(b, 29)) continue;
				if (check_sieve(b, 73)) continue;

				if (check_sieve(b, 11)) continue;
				if (check_sieve(b, 1409)) continue;
				if (check_sieve(b, 353)) continue;

				if (check_sieve(b, 37)) continue;
				if (check_sieve(b, 89)) continue;
				if (check_sieve(b, 241)) continue;

				if (check_sieve(b, 53)) continue;
				if (check_sieve(b, 19)) continue;
				if (check_sieve(b, 1217)) continue;

				if (check_sieve(b, 137)) continue;
				if (check_sieve(b, 61)) continue;
				if (check_sieve(b, 2689)) continue;

				if (check_sieve(b, 673)) continue;
				if (check_sieve(b, 337)) continue;
				if (check_sieve(b, 23)) continue;

				if (check_sieve(b, 1601)) continue;
				if (check_sieve(b, 3329)) continue;
				if (check_sieve(b, 401)) continue;

				if (check_sieve(b, 3457)) continue;
				if (check_sieve(b, 433)) continue;
				if (check_sieve(b, 929)) continue;

				if (check_sieve(b, 31)) continue;
				if (check_sieve(b, 233)) continue;
				if (check_sieve(b, 2113)) continue;

				if (check_sieve(b, 101)) continue;
				if (check_sieve(b, 4481)) continue;
				if (check_sieve(b, 109)) continue;

				if (check_sieve(b, 4993)) continue;
				if (check_sieve(b, 593)) continue;
				if (check_sieve(b, 281)) continue;

				if (check_sieve(b, 1249)) continue;
				if (check_sieve(b, 43)) continue;
				if (check_sieve(b, 2753)) continue;

				if (check_sieve(b, 313)) continue;
				if (check_sieve(b, 47)) continue;
				if (check_sieve(b, 149)) continue;

				if (check_sieve(b, 3137)) continue;
				if (check_sieve(b, 6529)) continue;
				if (check_sieve(b, 157)) continue;

				if (check_sieve(b, 1697)) continue;
				if (check_sieve(b, 7297)) continue;
				if (check_sieve(b, 173)) continue;

				if (check_sieve(b, 409)) continue;
				if (check_sieve(b, 881)) continue;
				if (check_sieve(b, 59)) continue;

				if (check_sieve(b, 181)) continue;
				if (check_sieve(b, 7681)) continue;
				if (check_sieve(b, 1889)) continue;

				if ((b & (b - 1)) == 0) continue;	// power of two
				if (!prp(b + 1)) continue;

				mpz_set_ui(b2n, 1); b2n->_mp_d[0] = b;

				int n = -1;
				do
				{
					++n;
					if (n == 6) break;
					mpz_mul(b2n, b2n, b2n);
					mpz_add_ui(r, b2n, 1);
					mpz_powm(r, two, b2n, r);
				} while (mpz_cmp_ui(r, 1) == 0);	// 2-prp

				if (n >= 5)
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

					if (n >= 6) output(b, n + 1);
				}
			}

			mpz_clear(b2n); mpz_clear(r);
		}
#ifdef VALID
		break;
#endif
	}

	mpz_clear(two);

	return EXIT_SUCCESS;
}

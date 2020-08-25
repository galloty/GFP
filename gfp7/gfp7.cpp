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
	ss << "Usage: gfp7 [b_min] [b_max] [numThreads]" << std::endl;
	ss << "         b_min is the start of the b search (in T (10^12) values, default 0)" << std::endl;
	ss << "         b_max is the end of the b search (in T (10^12) values, default 2^61)" << std::endl;
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
#define check_sieve(b, P)	if (sieve_##P[b % P] != 0) continue;

#define check_sv(P, ind)	(sieve_##P[b_p[ind]] != 0)

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
	const int b_max = (argc > 2) ? std::atoi(argv[2]) : 2305844;
	const int numThreads = (argc > 3) ? std::atoi(argv[3]) : 0;

#ifdef VALID
	static const size_t test_size = 102;
	static const uint64_t b_test[test_size] = { 2072005925466ull, 5082584069416ull, 12698082064890ull, 29990491969260ull, 46636691707050ull,
		65081025897426ull, 83689703895606ull, 83953213480290ull, 105003537341346ull, 105699143244090ull, 107581715369910ull, 111370557491826ull,
		111587899569066ull, 128282713771996ull, 133103004825210ull, 154438435790340ull, 161027983349016ull, 169081476836850ull, 199763736975426ull,
		201852155494656ull, 261915314595636ull, 268118920455760ull, 286556471193256ull, 301118628461886ull, 312235790115726ull, 324131153489776ull,
		355306576579540ull, 359868138215170ull, 360451719089070ull, 382897414818370ull, 400403462797566ull, 405113387071246ull, 463603317696946ull,
		473260661898880ull, 476479174179736ull, 482872515565840ull, 487843898732940ull, 529270045932306ull, 529904637878730ull, 533849609566396ull,
		593750085647950ull, 634749607136296ull, 638895561059136ull, 674246937239076ull, 675332667126276ull, 681789036366390ull, 687515678867046ull,
		702469331078776ull, 727781457982660ull, 732245491589670ull, 740290229100376ull, 796774361853936ull, 811756826917290ull, 831001337306046ull,
		913553120301760ull, 944402689383450ull, 944775163147420ull, 951004344430270ull, 976823130565266ull, 994296938432640ull, 995671839279390ull,
		1020706067098546ull, 1093658829642640ull, 1096351918526190ull, 1099329289783890ull, 1107514875580390ull, 1111034610603916ull, 1111213431100666ull,
		1115157431578180ull, 1119778188421506ull, 1125555835769746ull, 1139169960914196ull, 1150373034601546ull, 1200502751305050ull, 1220163152863540ull,
		1252928295612550ull, 1258270912111860ull, 1269970387252116ull, 1284354090928230ull, 1317985720932826ull, 1366538000684656ull, 1386703406823016ull,
		1407737085378906ull, 1423914424896870ull, 1496534979492946ull, 1518167728364446ull, 1562774978919616ull, 1574931481242970ull, 1575507399778560ull,
		1582797219258946ull, 1590635499964266ull, 1633253103667650ull, 1641761597988120ull, 1672445510479506ull, 1675441725971596ull, 1682620143003856ull,
		1689739603623270ull, 1691312542137946ull, 1759896904888596ull, 1809165672440640ull, 1888901949725830ull,
		240164550712338756ull };
#endif

	// weights: 17: 0.117647, 5: 0.4, 2: 0.5, 3: 0.666667
	// The pattern is 0, 120, 136, 256, 306, 340, 426, 460 (mod 510 = 2 * 3 * 5 * 17)
	static const size_t pattern_size = 8;
	static const size_t pattern_mod = 510;
	static const uint16_t pattern_step[pattern_size] = { 120, 16, 120, 50, 34, 86, 34, 50 };

	const size_t vsize = 16;
	typedef uint16_t vec[vsize] __attribute__((aligned(32)));							// xmm or ymm registers
	// weights: 257: 0.505837, 7 * 97: 0.583210 (97: 0.680412, 7: 0.857143), 13 * 41: 0.6378986 (13: 0.769231, 41: 0.829268)
	// weights: 193: 0.673575, 641: 0.801872, 769: 0.83485, 449: 0.859688, 113: 0.867257, 1153: 0.889853, 577: 0.890815
	// weights: 29: 0.896552, 73: 0.90411, 11: 0.909091, 1409: 0.909865, 353: 0.912181, 37: 0.918919
	static const vec step_p = { 257, 7 * 97, 13 * 41, 193, 641, 769, 449, 113, 1153, 577, 29, 73, 11, 1409, 353, 37 };
	static vec pattern_step_p[pattern_size];
	for (size_t j = 0; j < pattern_size; ++j)
	{
		for (size_t i = 0; i < vsize; ++i) pattern_step_p[j][i] = pattern_step[j] % step_p[i];
	}

	def_sieve(257);
	static uint8_t sieve_7_97[7 * 97], sieve_13_41[13 * 41]; sieveP2(sieve_7_97, 7, 97); sieveP2(sieve_13_41, 13, 41);
	def_sieve(193); def_sieve(641); def_sieve(769); def_sieve(449); def_sieve(113); def_sieve(1153); def_sieve(577);
	def_sieve(29); def_sieve(73); def_sieve(11); def_sieve(1409); def_sieve(353); def_sieve(37);

	def_sieve(89); def_sieve(241); def_sieve(53); def_sieve(19); def_sieve(1217); def_sieve(137); def_sieve(61);
	def_sieve(2689); def_sieve(673); def_sieve(337); def_sieve(23); def_sieve(1601); def_sieve(3329); def_sieve(401);
	def_sieve(3457); def_sieve(433); def_sieve(929); def_sieve(31); def_sieve(233); def_sieve(2113); def_sieve(101);
	def_sieve(4481); def_sieve(109); def_sieve(4993); def_sieve(593); def_sieve(281); def_sieve(1249); def_sieve(43);
	def_sieve(2753); def_sieve(313); def_sieve(47); def_sieve(149); def_sieve(3137); def_sieve(6529); def_sieve(157);
	def_sieve(1697); def_sieve(7297); def_sieve(173); def_sieve(409); def_sieve(881); def_sieve(59); def_sieve(181);
	def_sieve(7681); def_sieve(1889); def_sieve(7937); def_sieve(2017); def_sieve(977); def_sieve(457); def_sieve(197);
	def_sieve(67); def_sieve(2081); def_sieve(1009); def_sieve(4289); def_sieve(71); def_sieve(2273); def_sieve(4673);
	def_sieve(521); def_sieve(9473); def_sieve(9601); def_sieve(4801); def_sieve(229); def_sieve(9857); def_sieve(79);
	def_sieve(1201); def_sieve(569); def_sieve(10369); def_sieve(83); def_sieve(2593); def_sieve(10753); def_sieve(2657);
	def_sieve(601); def_sieve(5441); def_sieve(1297); def_sieve(617); def_sieve(5569); def_sieve(269); def_sieve(11393);
	def_sieve(1361); def_sieve(277); def_sieve(11777); def_sieve(5953); def_sieve(12161); def_sieve(12289); def_sieve(293);
	def_sieve(3041); def_sieve(1489); def_sieve(6337); def_sieve(3169); def_sieve(103); def_sieve(1553); def_sieve(13313);
	def_sieve(317); def_sieve(13441); def_sieve(107); def_sieve(13697); def_sieve(3361); def_sieve(761); def_sieve(6977);
	def_sieve(14081); def_sieve(14593); def_sieve(809); def_sieve(349); def_sieve(3617); def_sieve(1777); def_sieve(7489);
	def_sieve(15233); def_sieve(15361); def_sieve(857); def_sieve(373); def_sieve(1873); def_sieve(7873); def_sieve(16001);
	def_sieve(127); def_sieve(4001); def_sieve(389); def_sieve(131); def_sieve(397); def_sieve(4129); def_sieve(937);
	def_sieve(8513); def_sieve(953); def_sieve(8641); def_sieve(139); def_sieve(421); def_sieve(17921); def_sieve(2129);
	def_sieve(18049); def_sieve(2161); def_sieve(18433); def_sieve(4513); def_sieve(9281); def_sieve(1033); def_sieve(1049);
	def_sieve(19073); def_sieve(151); def_sieve(19457); def_sieve(461); def_sieve(19841); def_sieve(1097); def_sieve(20353);

	mpz_t two; mpz_init_set_ui(two, 2);

	uint64_t b_start = uint64_t(1e12 * b_min), b_end = uint64_t(1e12 * b_max) + 1;

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

	const size_t slice = size_t(1) << 25;

	uint64_t b_ctx = 0;
	std::ifstream ctxFile("gfp7.ctx");
	if (ctxFile.is_open())
	{
		ctxFile >> b_ctx;
		ctxFile.close();
	}

	const bool resume = (b_start < b_ctx);
	if (resume) b_start = b_ctx;
	b_start /= pattern_mod; b_start *= pattern_mod;
	b_end /= pattern_mod; b_end *= pattern_mod; b_end += slice * pattern_mod * n_thread;
	if ((b_end >> 61) != 0) b_end = uint64_t(1) << 61;

	std::cout << (resume ? "Resuming from a checkpoint, t" : "T") << "esting from " << b_start << " to " << b_end << std::endl;

	if (b_start == 0) output(1, 7);

	const timer::time start_time = timer::currentTime();

	for (uint64_t b_g = b_start; b_g < b_end; b_g += slice * pattern_mod * n_thread)
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
			vec b_p; for (size_t k = 0; k < vsize; ++k) b_p[k] = uint16_t(b % step_p[k]);

			mpz_t b2n, r; mpz_inits(b2n, r, nullptr);

			for (size_t i = 0, i_pattern = 0; i < slice * pattern_size; ++i, i_pattern = (i_pattern + 1) % pattern_size)
			{
				const vec & psi = pattern_step_p[i_pattern];

#pragma omp simd aligned(b_p, psi, step_p : 16)		// generates SSE2 or AVX2 instructions
				for (size_t k = 0; k < vsize; ++k)
				{
					const uint16_t r = b_p[k] + psi[k];
					const uint16_t p = step_p[k];
					b_p[k] = (r >= p) ? r - p : r;
				}

				b += pattern_step[i_pattern];

#ifdef VALID
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
				if (!bpat)
				{
					std::cout << "pattern error" << std::endl;
					break;
				}
				std::cout << i + 1 << ": ";
#endif
				if (check_sv(257, 0) | check_sv(7_97, 1) | check_sv(13_41, 2) | check_sv(193, 3) | check_sv(641, 4)) continue;
				if (check_sv(769, 5) | check_sv(449, 6) | check_sv(113, 7) | check_sv(1153, 8) | check_sv(577, 9)) continue;
				if (check_sv(29, 10) | check_sv(73, 11) | check_sv(11, 12) | check_sv(1409, 13) | check_sv(353, 14) | check_sv(37, 15)) continue;

				check_sieve(b, 89); check_sieve(b, 241); check_sieve(b, 53); check_sieve(b, 19); check_sieve(b, 1217); check_sieve(b, 137); check_sieve(b, 61);
				check_sieve(b, 2689); check_sieve(b, 673); check_sieve(b, 337); check_sieve(b, 23); check_sieve(b, 1601); check_sieve(b, 3329); check_sieve(b, 401);
				check_sieve(b, 3457); check_sieve(b, 433); check_sieve(b, 929); check_sieve(b, 31); check_sieve(b, 233); check_sieve(b, 2113); check_sieve(b, 101);
				check_sieve(b, 4481); check_sieve(b, 109); check_sieve(b, 4993); check_sieve(b, 593); check_sieve(b, 281); check_sieve(b, 1249); check_sieve(b, 43);
				check_sieve(b, 2753); check_sieve(b, 313); check_sieve(b, 47); check_sieve(b, 149); check_sieve(b, 3137); check_sieve(b, 6529); check_sieve(b, 157);
				check_sieve(b, 1697); check_sieve(b, 7297); check_sieve(b, 173); check_sieve(b, 409); check_sieve(b, 881); check_sieve(b, 59); check_sieve(b, 181);
				check_sieve(b, 7681); check_sieve(b, 1889); check_sieve(b, 7937); check_sieve(b, 2017); check_sieve(b, 977); check_sieve(b, 457); check_sieve(b, 197);

				if ((b & (b - 1)) == 0) continue;	// power of two
				if (!prp(b + 1)) continue;

				check_sieve(b, 67); check_sieve(b, 2081); check_sieve(b, 1009); check_sieve(b, 4289); check_sieve(b, 71); check_sieve(b, 2273); check_sieve(b, 4673);
				check_sieve(b, 521); check_sieve(b, 9473); check_sieve(b, 9601); check_sieve(b, 4801); check_sieve(b, 229); check_sieve(b, 9857); check_sieve(b, 79);
				check_sieve(b, 1201); check_sieve(b, 569); check_sieve(b, 10369); check_sieve(b, 83); check_sieve(b, 2593); check_sieve(b, 10753); check_sieve(b, 2657);
				check_sieve(b, 601); check_sieve(b, 5441); check_sieve(b, 1297); check_sieve(b, 617); check_sieve(b, 5569); check_sieve(b, 269); check_sieve(b, 11393);
				check_sieve(b, 1361); check_sieve(b, 277); check_sieve(b, 11777); check_sieve(b, 5953); check_sieve(b, 12161); check_sieve(b, 12289); check_sieve(b, 293);
				check_sieve(b, 3041); check_sieve(b, 1489); check_sieve(b, 6337); check_sieve(b, 3169); check_sieve(b, 103); check_sieve(b, 1553); check_sieve(b, 13313);
				check_sieve(b, 317); check_sieve(b, 13441); check_sieve(b, 107); check_sieve(b, 13697); check_sieve(b, 3361); check_sieve(b, 761); check_sieve(b, 6977);
				check_sieve(b, 14081); check_sieve(b, 14593); check_sieve(b, 809); check_sieve(b, 349); check_sieve(b, 3617); check_sieve(b, 1777); check_sieve(b, 7489);
				check_sieve(b, 15233); check_sieve(b, 15361); check_sieve(b, 857); check_sieve(b, 373); check_sieve(b, 1873); check_sieve(b, 7873); check_sieve(b, 16001);
				check_sieve(b, 127); check_sieve(b, 4001); check_sieve(b, 389); check_sieve(b, 131); check_sieve(b, 397); check_sieve(b, 4129); check_sieve(b, 937);
				check_sieve(b, 8513); check_sieve(b, 953); check_sieve(b, 8641); check_sieve(b, 139); check_sieve(b, 421); check_sieve(b, 17921); check_sieve(b, 2129);
				check_sieve(b, 18049); check_sieve(b, 2161); check_sieve(b, 18433); check_sieve(b, 4513); check_sieve(b, 9281); check_sieve(b, 1033); check_sieve(b, 1049);
				check_sieve(b, 19073); check_sieve(b, 151); check_sieve(b, 19457); check_sieve(b, 461); check_sieve(b, 19841); check_sieve(b, 1097); check_sieve(b, 20353);

				mpz_set_ui(b2n, 1); b2n->_mp_d[0] = b;

				const int n_min = 7;
				int n = 1;
				mpz_mul(b2n, b2n, b2n);
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
					mpz_set_ui(b2n, 1); b2n->_mp_d[0] = b;

					n = 0;
					while (true)
					{
						mpz_add_ui(r, b2n, 1);
						if (mpz_probab_prime_p(r, 10) == 0) break;
						mpz_mul(b2n, b2n, b2n);
						++n;
					}

					if (n >= n_min) output(b, n);
				}
			}

			mpz_clears(b2n, r, nullptr);
		}
#ifdef VALID
		break;
#endif
	}

	mpz_clear(two);

	return EXIT_SUCCESS;
}

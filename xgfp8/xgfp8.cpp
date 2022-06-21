/*
Copyright 2022, Yves Gallot

xgfp8 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.

xgfp8 searches for extended Generalized Fermat Progressions with length >= 8: numbers a, b such that a^{2^k} + b^{2^k} are primes for k = 0...7.
The integer sequence is https://oeis.org/A343121.
*/

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <vector>
#include <functional>

#include <gmp.h>
#include <omp.h>

// #define	VALID
// #define	DISP_RATIO

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
	ss << "xgfp8 0.2.0 " << sysver << ssc.str() << std::endl;
	ss << "Copyright (c) 2022, Yves Gallot" << std::endl;
	ss << "xgfp8 is free source code, under the MIT license." << std::endl;
	ss << std::endl;
	return ss.str();
}

static std::string usage()
{
	std::ostringstream ss;
	ss << "Usage: xgfp8 <a_min> <a_max> <numThreads>" << std::endl;
	ss << "   search in [a_min; a_max] (default a_min = 2, a_max = 2^32 - 1)" << std::endl;
	ss << "   numThreads is the number of threads (default 0 = one thread per logical CPU)" << std::endl << std::endl;
	return ss.str();
}

static void output(const uint32_t a, const uint32_t b, const int n)
{
	std::cout << "                               \r" << a << ", " << b << "\t" << "xGFP-" << n << std::endl;

	if (n >= 8)
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

inline uint32_t addmod32(const uint32_t x, const uint32_t y, const uint32_t m)
{
	const uint32_t c = (x >= m - y) ? m : 0;
	return x + y - c;
}

inline uint64_t addmod64(const uint64_t x, const uint64_t y, const uint64_t m)
{
	const uint64_t c = (x >= m - y) ? m : 0;
	return x + y - c;
}

inline uint64_t mul_hi(const uint64_t x, const uint64_t y) { return uint64_t((x * __uint128_t(y)) >> 64); }


class GFP
{
private:
	static constexpr uint32_t mod[] = {
		3 * 5 * 17, 7 * 97, 13 * 41, 769, 193, 641, 11 * 37, 23 * 29, 449, 113, 1153, 577, 73, 1409, 353, 19 * 31, 89,
		241, 53, 1217, 137, 61, 673, 337, /* 401, 433, 929, 233, 101, 109, 593, 281, 1249, 43 */ };
	static const size_t mod_size = sizeof(mod) / sizeof(uint32_t);

	mpz_t two, va2n[256], vb2n[256], vxgfn[256], vr[256];
	const size_t nthread;
	bool * const sieve_array;

protected:
	static int ilog2(const uint64_t x) { return 63 - __builtin_clzll(x); }

	// Barrett's product: let n = 63, r = ceil(log2(p)), p_shift = r - 2 = ceil(log2(p)) - 1, t = n + 1 = 64,
	// p_inv = floor(2^(s + t) / p). Then the number of iterations h = 1.
	// We must have x^2 < alpha.p with alpha = 2^(n-2). If p <= 2^(n-2) = 2^61 then x^2 < p^2 <= alpha.p.

	static uint64_t barrett_inv(const uint64_t p, int & p_shift)
	{
		p_shift = ilog2(p) - 1;
		return uint64_t((__uint128_t(1) << (p_shift + 64)) / p);
	}

	static uint64_t barrett_mul(const uint64_t x, const uint64_t y, const uint64_t p, const uint64_t p_inv, const int p_shift)
	{
		const __uint128_t xy = x * __uint128_t(y);
		uint64_t r = uint64_t(xy) - mul_hi(uint64_t(xy >> p_shift), p_inv) * p;
		if (r >= p) r -= p;
		return r;
	}

	static bool prp(const uint64_t n)	// n must be odd, 2-prp test
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

	static uint32_t gcd(const uint32_t x, const uint32_t y)
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
				for (size_t n = 1; n <= 8 - 1; ++n)
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

	bool check_sieve(const uint32_t * const a_mod_mul, const uint32_t b, const uint32_t b_255) const
	{
		const bool * psieve = sieve_array;

		if (psieve[a_mod_mul[0] + b_255] != 0) return true;
		psieve += 255 * 255;

#pragma GCC unroll 100
		for (size_t i = 1; i < mod_size; ++i)
		{
			const uint32_t p = mod[i];
			if (psieve[a_mod_mul[i] + (b % p)] != 0) return true;
			psieve += p * p;
		}
		return false;
	}

	void check_pseq(const size_t thread_id, const uint32_t a, const uint32_t b)
	{
		if (!prp(a + uint64_t(b))) return;

		mpz_t & a2n = va2n[thread_id]; mpz_t & b2n = vb2n[thread_id];
		mpz_t & xgfn = vxgfn[thread_id]; mpz_t & r = vr[thread_id];
	
		mpz_set_ui(a2n, a); mpz_set_ui(b2n, b);

		const int n_min = 8 - 1;
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

public:
	GFP(const size_t n_thread) : nthread(n_thread), sieve_array(new bool[421877741])
	{
		bool * psieve = sieve_array;

		for (size_t i = 0; i < mod_size; ++i)
		{
			const uint32_t & p = mod[i];
			fill_sieve(psieve, p);
			psieve += p * p;
		}
		std::cout << "sieve_array size: " <<  psieve - sieve_array << std::endl;

		mpz_init_set_ui(two, 2);
		for (size_t i = 0; i < nthread; ++i) mpz_inits(va2n[i], vb2n[i], vxgfn[i], vr[i], nullptr);
	}

	virtual ~GFP()
	{
		delete[] sieve_array;
		for (size_t i = 0; i < nthread; ++i) mpz_clears(va2n[i], vb2n[i], vxgfn[i], vr[i], nullptr);
	}

	void gen_sieve() const
	{
		std::vector<std::pair<uint32_t, double>> weights;

		const uint32_t p_max = 1500;
		bool * const sieve = new bool[p_max * size_t(p_max)];
		for (uint32_t p = 3; p <= p_max; p += 2)
		{
			bool prm = true;
			for (uint32_t d = 3; d < p; d += 2) if (p % d == 0) { prm = false; break; }
			if (!prm) continue;
			const double weight = fill_sieve(sieve, p) / (p * double(p));
			weights.push_back(std::make_pair(p, weight));
			std::cout << p << ": " << weight << std::endl;
		}
		delete[] sieve;

		std::sort(weights.begin(), weights.end(), [&](const auto & p1, const auto & p2) { return p1.second < p2.second; });

		const size_t count = 100;
		for (size_t i = 0; i < count; ++i) std::cout << weights[i].first << ": " << weights[i].second << std::endl;
	}

	void check(const uint32_t a_start_257, const uint32_t a_end_257)
	{
#ifdef GEN_SIEVE
		gen_sieve();
		return;
#endif

		timer::time disp_time = timer::currentTime();
		uint32_t disp_a_257 = a_start_257;
#ifdef DISP_RATIO
		size_t wcount = 0, scount = 0;
#endif

		for (uint32_t a_257 = a_start_257; a_257 <= a_end_257; a_257 += 16 * nthread)
		{
			const timer::time cur_time = timer::currentTime();
			const double dt = timer::diffTime(cur_time, disp_time);
			if (dt > 10)
			{
				disp_time = cur_time;
				const double da = (a_257 - disp_a_257) / dt;
				std::cout << a_257 * 257 << ", +" << std::setprecision(3) << da << ", " << 1e-6 * 86400 * da * 257 << "M/day";
#ifdef DISP_RATIO
				std::cout << ", 1/" << wcount / scount;
				wcount = scount = 0;
#endif
				std::cout << "       \r" << std::flush;
				disp_a_257 = a_257;

				std::ofstream ctxFile("xgfp8.ctx");
				if (ctxFile.is_open())
				{
					ctxFile << a_257 * 257 << std::endl;
					ctxFile.flush();
					ctxFile.close();
				}
			}

#pragma omp parallel for schedule(dynamic)
			for (size_t j = 0; j < 16 * nthread; ++j)
			{
				const size_t thread_id = size_t(omp_get_thread_num());

				uint32_t a = (a_257 + j) * 257;
				uint32_t a_mod[mod_size], a_mod_mul[mod_size];
				for (size_t i = 0; i < mod_size; ++i)
				{
					const uint32_t m = mod[i], am = a % m;
					a_mod[i] = am; a_mod_mul[i] = am * m;
				}

				// a = 0 (mod 257) then check all 0 < b < a such that a + b != 0 (mod 2)
				for (uint32_t b = (a % 2) + 1, b_255 = b; b < a; b += 2, b_255 = addmod32(b_255, 2, 255))
				{
#ifdef DISP_RATIO
					++wcount;
#endif
					if (check_sieve(a_mod_mul, b, b_255)) continue;
#ifdef DISP_RATIO
					++scount;
#endif
					check_pseq(thread_id, a, b);
				}

				// a != 0 (mod 257) then check 0 < b < a such that b = 0 (mod 257) or b = a (mod 257) and a + b != 0 (mod 2)
				for (uint32_t i = 1; i < 257; ++i)
				{
					++a;
					for (size_t i = 0; i < mod_size; ++i)
					{
						const uint32_t m = mod[i], am = addmod32(a_mod[i], 1, m);
						a_mod[i] = am; a_mod_mul[i] = am * m;
					}

					uint32_t b0 = 257, b0_255 = b0 - 255;
					if ((a + b0) % 2 == 0) { b0 += 257; b0_255 = addmod32(b0_255, 257 - 255, 255); }
					for (uint32_t b = b0, b_255 = b0_255; b < a; b += 2 * 257, b_255 = addmod32(b_255, 2 * (257 - 255), 255))
					{
#ifdef DISP_RATIO
						++wcount;
#endif
						if (check_sieve(a_mod_mul, b, b_255)) continue;
#ifdef DISP_RATIO
						++scount;
#endif
						check_pseq(thread_id, a, b);
					}

					uint32_t ba = a % 257, ba_255 = (ba >= 255) ? ba - 255 : ba;
					if ((a + ba) % 2 == 0) { ba += 257; ba_255 = addmod32(ba_255, 257 - 255, 255); }

					for (uint32_t b = ba, b_255 = ba_255; b < a; b += 2 * 257, b_255 = addmod32(b_255, 2 * (257 - 255), 255))
					{
#ifdef DISP_RATIO
						++wcount;
#endif
						if (check_sieve(a_mod_mul, b, b_255)) continue;
#ifdef DISP_RATIO
						++scount;
#endif
						check_pseq(thread_id, a, b);
					}
				}
			}
		}
	}
};

#ifdef VALID
static void valid(const size_t n_thread)
{
	const size_t n = 36;
	static constexpr uint32_t a[n] = { 26507494, 56984867, 62055998, 63491771, 89616928, 113780846, 134733857, 139403406, 151032318, 152120099, 160853473,
		162552757, 164334410, 168637489, 182386475, 189919346, 190611395, 203833179, 206250862, 213384510, 217336419, 233509429, 241553272, 251554684,
		274484657, 279516296, 285124157, 290163473, 291329833, 298260240, 308235968, 314945196, 318675558, 328492003, 336810340, 337201010 };

	GFP gfp(n_thread);
	for (size_t i = 0; i < n; ++i)
	{
		std::cout << i + 1 << ": " << std::endl;
		const uint32_t a_257 = a[i] / 257;
		gfp.check(a_257, a_257);
	}
}
#endif

int main(int argc, char * argv[])
{
	std::cout << header();
	std::cout << usage();

	const uint32_t a_min = (argc > 1) ? uint32_t(std::atoll(argv[1])) : 2;
	const uint32_t a_max = (argc > 2) ? uint32_t(std::atoll(argv[2])) : uint32_t(-1);
	const int numThreads = (argc > 3) ? std::atoi(argv[3]) : 0;

	if (numThreads != 0) omp_set_num_threads(numThreads);

	size_t n_thread = 1;
#pragma omp parallel
{
	n_thread = omp_get_num_threads();
}
	std::cout << n_thread << " thread(s)" << std::endl;

#ifdef VALID
	valid(n_thread);
	return EXIT_SUCCESS;
#endif

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

	const uint32_t a_start_257 = a_start / 257, a_end_257 = a_end / 257 + ((a_end % 257 != 0) ? 1 : 0);

	std::cout << (resume ? "Resuming from a checkpoint, t" : "T") << "esting from " << (a_start < 2 ? 2 : a_start) << " to " << a_end << std::endl;

	GFP gfp(n_thread);
	gfp.check(a_start_257, a_end_257);

	// gfp.gen_sieve();

	return EXIT_SUCCESS;
}

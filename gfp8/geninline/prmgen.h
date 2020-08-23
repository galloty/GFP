/*
Copyright 2020, Yves Gallot

gfp8 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#define _USE_MATH_DEFINES
#include <cmath>

class PrmGen
{
	private: static const size_t log2pMax = 32;					// pMax = 2^log2pMax.
	private: static const size_t log2spMax = log2pMax / 2;		// sieve pMax = sqrt(pMax).
	private: static const size_t spMax = 1ull << log2spMax;
	private: static const size_t sieveSize = spMax / 2;			// sieve with an odd prime table.
	private: static const size_t oddPrimeCount = 6541;			// # odd primes with p < spMax.

	private: uint32_t * const prm;
	private: uint64_t * const prmPtr;
	private: uint8_t * const sieveTable;
	private: uint64_t jp, kp;

	private: void fillSieve()
	{
		for (size_t i = 0; i < oddPrimeCount; ++i)
		{
			uint64_t k;
			for (k = prmPtr[i]; k < sieveSize; k += prm[i]) sieveTable[k] = 1;
			prmPtr[i] = k - sieveSize;
		}
	}

	public: PrmGen() : prm(new uint32_t[oddPrimeCount]), prmPtr(new uint64_t[oddPrimeCount]), sieveTable(new uint8_t[sieveSize]), jp(0), kp(0)
	{
		prm[0] = 3; prm[1] = 5; prm[2] = 7;
		uint32_t i = 3;
		for (uint32_t k = 11; k < uint32_t(spMax); k += 2)
		{
			const uint32_t s = uint32_t(sqrt(double(k))) + 1;
			uint32_t d;
			for (d = 3; d <= s; d += 2) if (k % d == 0) break;
			if (d > s)
			{
				prm[i] = k;
				++i;
			}
		}

		if (i != oddPrimeCount) throw;
	}

	public: ~PrmGen()
	{
		delete [] sieveTable;
		delete [] prmPtr;
		delete [] prm;
	}

	public: uint64_t first()
	{
		for (size_t i = 0; i < oddPrimeCount; ++i) prmPtr[i] = (prm[i] >> 1) + (uint64_t)prm[i];

		for (size_t k = 0; k < sieveSize; ++k) sieveTable[k] = 0;

		fillSieve();

		jp = 0;
		kp = 1;

		return 2;
	}

	public: uint64_t next()
	{
		do
		{
			while (kp < sieveSize)
			{
				if (sieveTable[kp] == 0)
				{
					const uint64_t p = (jp << log2spMax) + 2 * kp + 1;
					++kp;
					return p;
				}
				else
				{
					sieveTable[kp] = 0;
					++kp;
				}
			}

			fillSieve();
			kp = 0;
			++jp;
		}
		while (jp < spMax);

		jp = 0;
		return 0;
	}
};

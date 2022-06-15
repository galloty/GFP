/*
Copyright 2022, Yves Gallot

xgfp8 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <cmath>

// Segmented sieve of Eratosthenes
class PrmGen
{
private:
	static const uint32_t sp_max = 1 << 16;
	static const size_t sieve_size = sp_max / 2;	// sieve with an odd prime table.
	static const size_t odd_prime_count = 6541;		// # odd primes with p < 2^16.

	bool _sieve[sieve_size];
	uint32_t _prm[odd_prime_count];
	uint32_t _prm_ptr[odd_prime_count];

	uint32_t _jp, _kp;

private:
	void fill_sieve()
	{
		for (size_t k = 0; k < sieve_size; ++k) _sieve[k] = false;

		for (size_t i = 0; i < odd_prime_count; ++i)
		{
			size_t k = _prm_ptr[i], p = _prm[i];
			for (; k < sieve_size; k += p) _sieve[k] = true;
			_prm_ptr[i] = uint32_t(k - sieve_size);
		}
	}

public:
	PrmGen() : _jp(0), _kp(0)
	{
		_prm[0] = 3; _prm[1] = 5; _prm[2] = 7;
		uint32_t i = 3;
		for (uint32_t k = 11; k < sp_max; k += 2)
		{
			const uint32_t s = uint32_t(std::sqrt(double(k))) + 1;
			uint32_t d; for (d = 3; d <= s; d += 2) if (k % d == 0) break;
			if (d > s) _prm[i++] = k;
		}
	}

	uint32_t first()
	{
		for (size_t i = 0; i < odd_prime_count; ++i) _prm_ptr[i] = (_prm[i] >> 1) + _prm[i];

		fill_sieve();
		_jp = 0; _kp = 1;
		return 2;
	}

	uint32_t next()
	{
		while (true)
		{
			while (_kp < sieve_size)
			{
				const uint32_t k = _kp++;
				if (!_sieve[k]) return _jp + 2 * k + 1;
			}

			fill_sieve();
			_jp += sp_max; _kp = 0;
		}
	}
};

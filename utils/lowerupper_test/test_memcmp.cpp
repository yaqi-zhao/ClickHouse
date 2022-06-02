#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <math.h>
#include <cstdint>
#include <algorithm>

using namespace std;
using namespace std::chrono;

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include <immintrin.h>

// using UInt8 = ::UInt8;
// using Char = UInt8;

namespace detail
{

template <typename T>
inline int cmp(T a, T b)
{
    if (a < b)
        return -1;
    if (a > b)
        return 1;
    return 0;
}

}

static __inline uint64_t
rdtsc(void)
{
	uint32_t low, high;

	__asm __volatile("rdtsc" : "=a" (low), "=d" (high));
	return (low | ((uint64_t)high << 32));
}


int memcmpSmallAllowOverflow15_AVX512(const char * a, size_t a_size, const char * b, size_t b_size)
{
    size_t min_size = std::min(a_size, b_size);

    for (size_t offset = 0; offset < min_size; offset += 16)
    {
        uint16_t mask = _mm_cmp_epi8_mask(
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + offset)),
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + offset)), _MM_CMPINT_NE);

        if (mask)
        {
            offset += __builtin_ctz(mask);

            if (offset >= min_size)
                break;

            return detail::cmp(a[offset], b[offset]);
        }
    }

    return detail::cmp(a_size, b_size);
}

int memcmpSmallAllowOverflow15_SSE2(const char * a, size_t a_size, const char * b, size_t b_size)
{
    size_t min_size = std::min(a_size, b_size);

    for (size_t offset = 0; offset < min_size; offset += 16)
    {
        uint16_t mask = _mm_movemask_epi8(_mm_cmpeq_epi8(
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + offset)),
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + offset))));
        mask = ~mask;

        if (mask)
        {
            offset += __builtin_ctz(mask);

            if (offset >= min_size)
                break;

            return detail::cmp(a[offset], b[offset]);
        }
    }

    return detail::cmp(a_size, b_size);
}

void run_test(const char * src, size_t src_len, char * dst, size_t len) {
    cout << "len: " << src_len << "/" << len << endl;

    auto start1 = rdtsc();
    for (int i = 0; i < 100; i++) {
        memcmpSmallAllowOverflow15_SSE2(src, src_len, dst, len);
    }
    auto end1 = rdtsc();
    // cout << dst << endl;
    auto duration1 = (end1 - start1) / 100;
    cout << "SSE2 cpu cycle: " << duration1  << endl;

    start1 = rdtsc();
    for (int i = 0; i < 100; i++) {
        memcmpSmallAllowOverflow15_AVX512(src, src_len, dst, len);
    }
    end1 = rdtsc();
    // cout << dst << endl;
    duration1 = (end1 - start1) / 100;
    cout << "AVX512 cpu cycle: " << duration1  << endl;
}

int main() {
//   char src[257] = {'\0'};
//   char dst[257] = {'\0'};

//   for (int i = 0; i < 16; i++) {
//     for (int j = 0; j < 16; j++) {
//       src[i * 16 + j] = 'a' + j;
//     }
//   }


//     for (int k = 0; k < 7; k++) {
//         size_t len = 16 * pow(2, k);
//         char src[len] = {'\0'};
//         char dst[len] = {'\0'};
//         for (int i = 0; i < pow(2, k); i++){
//             for (int j = 0; j < 16; j++) {
//                 src[i * 16 + j] = 'a' + j;
//             }
//         }
//         run_test(src, len, src, len);
//     }

    int a_length = 15;
    int b_length = 7;
    for (int k = 0; k < 7; k++) {
        size_t len = a_length * pow(2, k);
        size_t b_len = b_length * pow(2, k);
        char src[len] = {'\0'};
        char dst[b_len] = {'\0'};
        for (int i = 0; i < pow(2, k); i++){
            for (int j = 0; j < a_length; j++) {
                src[i * a_length + j] = 'a' + j;
            }
        }
        for (int i = 0; i < pow(2, k); i++) {
            for (int j = 0; j < b_length; j++) {
                dst[i * b_length + j] = 'a' + j;
            }
        }
        run_test(src, len, dst, b_len);
    }
}
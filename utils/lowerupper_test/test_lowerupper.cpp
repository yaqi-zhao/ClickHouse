#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <math.h>
using namespace std;
using namespace std::chrono;

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include <immintrin.h>

template <char not_case_lower_bound, char not_case_upper_bound>
struct LowerUpperImpl {
  public:
  static void arraySSE(const char * src, const char * src_end, char * dst) {
    const auto flip_case_mask = 'A' ^ 'a';

#ifdef __SSE2__
    const auto bytes_sse = sizeof(__m128i);
    const auto src_end_sse = src_end - (src_end - src) % bytes_sse;

    const auto v_not_case_lower_bound = _mm_set1_epi8(not_case_lower_bound - 1);
    const auto v_not_case_upper_bound = _mm_set1_epi8(not_case_upper_bound + 1);
    const auto v_flip_case_mask = _mm_set1_epi8(flip_case_mask);

    for (; src < src_end_sse; src += bytes_sse, dst += bytes_sse) {
      const auto chars = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));

      const auto is_not_case
              = _mm_and_si128(_mm_cmpgt_epi8(chars, v_not_case_lower_bound), _mm_cmplt_epi8(chars, v_not_case_upper_bound));

      const auto xor_mask = _mm_and_si128(v_flip_case_mask, is_not_case);

      const auto cased_chars = _mm_xor_si128(chars, xor_mask);

      _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), cased_chars);
    }
#endif

    for (; src < src_end; ++src, ++dst)
      if (*src >= not_case_lower_bound && *src <= not_case_upper_bound)
        *dst = *src ^ flip_case_mask;
      else
        *dst = *src;
  }

  static void arrayAVX512(const char * src, const char * src_end, char * dst) {
    const auto flip_case_mask = 'A' ^ 'a';

#ifdef __AVX512F__
    const auto byte_avx512 = sizeof(__m512i);
    const auto src_end_avx = src_end - (src_end - src) % byte_avx512;

    const auto v_not_case_lower_bound = _mm512_set1_epi64(not_case_lower_bound - 1);
    const auto v_not_case_upper_bound = _mm512_set1_epi64(not_case_upper_bound + 1);
    const auto v_flip_case_mask = _mm512_set1_epi64(flip_case_mask);

    for (; src < src_end_avx; src += byte_avx512, dst += byte_avx512) {
      const auto chars = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(src));

      const auto is_not_case
              = _mm512_and_si512(_mm512_movm_epi8(_mm512_cmpgt_epi8_mask(chars, v_not_case_lower_bound)), 
              _mm512_movm_epi8(_mm512_cmplt_epi8_mask(chars, v_not_case_upper_bound)));

      const auto xor_mask = _mm512_and_si512(v_flip_case_mask, is_not_case);

      const auto cased_chars = _mm512_xor_si512(chars, xor_mask);

      _mm512_storeu_si512(reinterpret_cast<__m512i *>(dst), cased_chars);
    }
#endif

    for (; src < src_end; ++src, ++dst)
      if (*src >= not_case_lower_bound && *src <= not_case_upper_bound)
        *dst = *src ^ flip_case_mask;
      else
        *dst = *src;
  }

  static void arrayAVX512SSE(const char * src, const char * src_end, char * dst) {
    const auto flip_case_mask = 'A' ^ 'a';

#ifdef __AVX512F__
    const auto byte_avx512 = sizeof(__m512i);
    const auto src_end_avx = src_end - (src_end - src) % byte_avx512;
    if (src < src_end_avx) {
        const auto v_not_case_lower_bound = _mm512_set1_epi8(not_case_lower_bound - 1);
        const auto v_not_case_upper_bound = _mm512_set1_epi8(not_case_upper_bound + 1);
        const auto v_flip_case_mask = _mm512_set1_epi8(flip_case_mask);

        for (; src < src_end_avx; src += byte_avx512, dst += byte_avx512) {
            const auto chars = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(src));

            const auto is_not_case
                    = _mm512_and_si512(_mm512_movm_epi8(_mm512_cmpgt_epi8_mask(chars, v_not_case_lower_bound)), 
                    _mm512_movm_epi8(_mm512_cmplt_epi8_mask(chars, v_not_case_upper_bound)));

            const auto xor_mask = _mm512_and_si512(v_flip_case_mask, is_not_case);

            const auto cased_chars = _mm512_xor_si512(chars, xor_mask);

            _mm512_storeu_si512(reinterpret_cast<__m512i *>(dst), cased_chars);
        }
    }
#endif

#ifdef __SSE2__
    const auto bytes_sse = sizeof(__m128i);
    const auto src_end_sse = src_end - (src_end - src) % bytes_sse;
    if (src < src_end_sse) {

        const auto v_not_case_lower_bound_1 = _mm_set1_epi8(not_case_lower_bound - 1);
        const auto v_not_case_upper_bound_1 = _mm_set1_epi8(not_case_upper_bound + 1);
        const auto v_flip_case_maskz_1 = _mm_set1_epi8(flip_case_mask);

        for (; src < src_end_sse; src += bytes_sse, dst += bytes_sse) {
        const auto chars = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));

        const auto is_not_case
                = _mm_and_si128(_mm_cmpgt_epi8(chars, v_not_case_lower_bound_1), _mm_cmplt_epi8(chars, v_not_case_upper_bound_1));

        const auto xor_mask = _mm_and_si128(v_flip_case_maskz_1, is_not_case);

        const auto cased_chars = _mm_xor_si128(chars, xor_mask);

        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), cased_chars);
        }
    }

#endif

    for (; src < src_end; ++src, ++dst)
      if (*src >= not_case_lower_bound && *src <= not_case_upper_bound)
        *dst = *src ^ flip_case_mask;
      else
        *dst = *src;
  }

  static void arrayNoSSE(const char * src, const char * src_end, char * dst) {
    const auto flip_case_mask = 'A' ^ 'a';
    for (; src < src_end; ++src, ++dst)
      if (*src >= not_case_lower_bound && *src <= not_case_upper_bound)
        *dst = *src ^ flip_case_mask;
      else
        *dst = *src;
  }
};


void run_test(const char * src, const char * src_end, char * dst, size_t len) {
    cout << "len: " << len << endl;
    LowerUpperImpl<'a', 'z'> lowerUpper;

    auto start1 = system_clock::now();
    for (int i = 0; i < 100; i++) {
    lowerUpper.arraySSE(src, src_end, dst);
    }
    auto end1 = system_clock::now();
    // cout << dst << endl;
    auto duration1 = duration_cast<nanoseconds>(end1 - start1);
    cout << "SSE2 Time cost: " << duration1.count() << " ns" << endl;

    auto start2 = system_clock::now();
    for (int i = 0; i < 100; i++) {
    lowerUpper.arrayNoSSE(src, src_end, dst);
    }
    auto end2 = system_clock::now();
    // cout << dst << endl;
    auto duration2 = duration_cast<nanoseconds>(end2 - start2);
    cout << "NO SSE2 Time cost: " << duration2.count() << " ns" << endl;

    auto start3 = system_clock::now();
    for (int i = 0; i < 100; i++) {
    lowerUpper.arrayAVX512(src, src_end, dst);
    }
    auto end3 = system_clock::now();
    // cout << dst << endl;
    auto duration3 = duration_cast<nanoseconds>(end3 - start3);
    cout << "AVX512 Time cost: " << duration3.count() << " ns" << endl;

    auto start4 = system_clock::now();
    for (int i = 0; i < 100; i++) {
    lowerUpper.arrayAVX512SSE(src, src_end, dst);
    }
    auto end4 = system_clock::now();
    // cout << dst << endl;
    auto duration4 = duration_cast<nanoseconds>(end4 - start4);
    cout << "AVX512 & SSE2 Time cost: " << duration4.count() << " ns" << endl;    
}

int main() {
  char src[257] = {'\0'};
  char dst[257] = {'\0'};

  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      src[i * 16 + j] = 'a' + j;
    }
  }


    for (int k = 0; k < 7; k++) {
        size_t len = 16 * pow(2, k);
        char src[len] = {'\0'};
        char dst[len] = {'\0'};
        for (int i = 0; i < pow(2, k); i++){
            for (int j = 0; j < 16; j++) {
                src[i * 16 + j] = 'a' + j;
            }
        }
        run_test(src, (src + len + 1), dst, len);
    }

    cout << "==============" << endl;
    for (int k = 1; k < 6; k++) {
        size_t len = pow(2, k) * 26;
        char src[len] = {'\0'};
        char dst[len] = {'\0'};
        for (int i = 0; i < pow(2, k); i++){
            for (int j = 0; j < 26; j++) {
                src[i * 26 + j] = 'a' + j;
            }
        }
        run_test(src, (src + len + 1), dst, len);
    }

//   LowerUpperImpl<'a', 'z'> lowerUpper;

//   auto start1 = system_clock::now();
//   for (int i = 0; i < 100; i++) {
//     lowerUpper.arraySSE(&src[0], &src[257], &dst[0]);
//   }
//   auto end1 = system_clock::now();
//   cout << dst << endl;
//   auto duration1 = duration_cast<nanoseconds>(end1 - start1);
//   cout << "SSE2 Time cost: " << duration1.count() << " ns" << endl;

//   auto start2 = system_clock::now();
//   for (int i = 0; i < 100; i++) {
//     lowerUpper.arrayNoSSE(&src[0], &src[257], &dst[0]);
//   }
//   auto end2 = system_clock::now();
//   cout << dst << endl;
//   auto duration2 = duration_cast<nanoseconds>(end2 - start2);
//   cout << "NO SSE2 Time cost: " << duration2.count() << " ns" << endl;

//   auto start3 = system_clock::now();
//   for (int i = 0; i < 100; i++) {
//     lowerUpper.arrayAVX512(&src[0], &src[257], &dst[0]);
//   }
//   auto end3 = system_clock::now();
//   cout << dst << endl;
//   auto duration3 = duration_cast<nanoseconds>(end3 - start3);
//   cout << "AVX512 Time cost: " << duration3.count() << " ns" << endl;

//   auto start4 = system_clock::now();
//   for (int i = 0; i < 100; i++) {
//     lowerUpper.arrayAVX512SSE(&src[0], &src[257], &dst[0]);
//   }
//   auto end4 = system_clock::now();
//   cout << dst << endl;
//   auto duration4 = duration_cast<nanoseconds>(end4 - start4);
//   cout << "AVX512 & SSE2 Time cost: " << duration4.count() << " ns" << endl;
}
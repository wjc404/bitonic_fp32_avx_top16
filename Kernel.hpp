#ifndef INCLUDE_BITONIC_AVX_F32_HEADER_
#define INCLUDE_BITONIC_AVX_F32_HEADER_

#include <immintrin.h>

/** the implementation of sort-16 graph from
 * blog.csdn.net/u010445006/article/details/74091378 */
template <typename SORT_IMPL>
void sort_16_f32_avx(__m256 &up, __m256 &down) {
  /** stage 1 */
  __m256 v1 = _mm256_shuffle_ps(up, down, 0xcc); // { 0, 3, 8, b, 4, 7, c, f }
  __m256 v2 = _mm256_shuffle_ps(up, down, 0x99); // { 1, 2, 9, a, 5, 6, d, e }
  __m256 vl = SORT_IMPL::pick_up(v1, v2);        // { 0, 3, 8, b, 4, 7, c, f }
  __m256 vs = SORT_IMPL::pick_down(v1, v2);      // { 1, 2, 9, a, 5, 6, d, e }
  /** stage 2 */
  v2 = _mm256_permute_ps(vs, 0xb1);              // { 2, 1, a, 9, 6, 5, e, d }
  vs = SORT_IMPL::pick_down(vl, v2);             // { 2, 3, a, b, 4, 5, c, d }
  vl = SORT_IMPL::pick_up(vl, v2);               // { 0, 1, 8, 9, 6, 7, e, f }
  v1 = _mm256_shuffle_ps(vl, vs, 0x88);          // { 0, 8, 2, a, 6, e, 4, c }
  v2 = _mm256_shuffle_ps(vl, vs, 0xdd);          // { 1, 9, 3, b, 7, f, 5, d }
  vl = SORT_IMPL::pick_up(v1, v2);               // { 0, 8, 2, a, 7, f, 5, d }
  vs = SORT_IMPL::pick_down(v1, v2);             // { 1, 9, 3, b, 6, e, 4, c }
  /** stage 3 */
  v1 = _mm256_permute_ps(vl, 0x4e);              // { 2, a, 0, 8, 5, d, 7, f }
  v2 = _mm256_permute2f128_ps(vs, vs, 1);        // { 6, e, 4, c, 1, 9, 3, b }
  vl = SORT_IMPL::pick_up(v1, v2);               // { 2, e, 0, c, 1, d, 3, f }
  vs = SORT_IMPL::pick_down(v1, v2);             // { 6, a, 4, 8, 5, 9, 7, b }
  v1 = _mm256_shuffle_ps(vl, vs, 0x44);          // { 2, e, 6, a, 1, d, 5, 9 }
  v2 = _mm256_shuffle_ps(vl, vs, 0xee);          // { 0, c, 4, 8, 3, f, 7, b }
  vl = SORT_IMPL::pick_up(v1, v2);               // { 0, e, 4, a, 1, f, 5, b }
  vs = SORT_IMPL::pick_down(v1, v2);             // { 2, c, 6, 8, 3, d, 7, 9 }
  v1 = _mm256_permute2f128_ps(vl, vs, 0x20);     // { 0, e, 4, a, 2, c, 6, 8 }
  v2 = _mm256_permute2f128_ps(vl, vs, 0x31);     // { 1, f, 5, b, 3, d, 7, 9 }
  vl = SORT_IMPL::pick_up(v1, v2);               // { 0, f, 4, b, 2, d, 6, 9 }
  vs = SORT_IMPL::pick_down(v1, v2);             // { 1, e, 5, a, 3, c, 7, 8 }
  /** stage 4 */
  v1 = _mm256_permute_ps(vl, 0x1b);              // { b, 4, f, 0, 9, 6, d, 2 }
  v2 = _mm256_permute2f128_ps(vs, vs, 1);        // { 3, c, 7, 8, 1, e, 5, a }
  vl = SORT_IMPL::pick_up(v1, v2);               // { 3, 4, 7, 0, 1, 6, 5, 2 }
  vs = SORT_IMPL::pick_down(v1, v2);             // { b, c, f, 8, 9, e, d, a }
  v1 = _mm256_unpacklo_ps(vl, vs);               // { 3, b, 4, c, 1, 9, 6, e }
  v2 = _mm256_unpackhi_ps(vl, vs);               // { 7, f, 0, 8, 5, d, 2, a }
  vl = SORT_IMPL::pick_up(v1, v2);               // { 3, b, 0, 8, 1, 9, 2, a }
  vs = SORT_IMPL::pick_down(v1, v2);             // { 7, f, 4, c, 5, d, 6, e }
  v1 = _mm256_permute2f128_ps(vl, vs, 0x20);     // { 3, b, 0, 8, 7, f, 4, c }
  v2 = _mm256_permute2f128_ps(vl, vs, 0x31);     // { 1, 9, 2, a, 5, d, 6, e }
  vl = SORT_IMPL::pick_up(v1, v2);               // { 1, 9, 0, 8, 5, d, 4, c }
  vs = SORT_IMPL::pick_down(v1, v2);             // { 3, b, 2, a, 7, f, 6, e }
  v1 = _mm256_unpacklo_ps(vl, vs);               // { 1, 3, 9, b, 5, 7, d, f }
  v2 = _mm256_unpackhi_ps(vl, vs);               // { 0, 2, 8, a, 4, 6, c, e }
  vl = SORT_IMPL::pick_up(v1, v2);               // { 0, 2, 8, a, 4, 6, c, e }
  vs = SORT_IMPL::pick_down(v1, v2);             // { 1, 3, 9, b, 5, 7, d, f }
  /** restore element order of results */
  up = _mm256_unpacklo_ps(vl, vs);               // { 0, 1, 2, 3, 4, 5, 6, 7 }
  down = _mm256_unpackhi_ps(vl, vs);             // { 8, 9, a, b, c, d, e, f }
}

/** the last stage in the sort graph mentioned above */
template <typename SORT_IMPL>
void sort_16_f32_avx_bitonic_input(__m256 &up, __m256 &down) {
  /** step 1 */
  __m256 vl = SORT_IMPL::pick_up(up, down);         // { 0, 1, 2, 3, 4, 5, 6, 7 }
  __m256 vs = SORT_IMPL::pick_down(up, down);       // { 8, 9, a, b, c, d, e, f }
  /** step 2 */
  __m256 v1 = _mm256_permute2f128_ps(vl, vs, 0x20); // { 0, 1, 2, 3, 8, 9, a, b }
  __m256 v2 = _mm256_permute2f128_ps(vl, vs, 0x31); // { 4, 5, 6, 7, c, d, e, f }
  vl = SORT_IMPL::pick_up(v1, v2);                  // { 0, 1, 2, 3, 8, 9, a, b }
  vs = SORT_IMPL::pick_down(v1, v2);                // { 4, 5, 6, 7, c, d, e, f }
  /** step 3 */
  v1 = _mm256_unpacklo_ps(vl, vs);                  // { 0, 4, 1, 5, 8, c, 9, d }
  v2 = _mm256_unpackhi_ps(vl, vs);                  // { 2, 6, 3, 7, a, e, b, f }
  vl = SORT_IMPL::pick_up(v1, v2);                  // { 0, 4, 1, 5, 8, c, 9, d }
  vs = SORT_IMPL::pick_down(v1, v2);                // { 2, 6, 3, 7, a, e, b, f }
  /** step 4 */
  v1 = _mm256_unpacklo_ps(vl, vs);                  // { 0, 2, 4, 6, 8, a, c, e }
  v2 = _mm256_unpackhi_ps(vl, vs);                  // { 1, 3, 5, 7, 9, b, d, f }
  vl = SORT_IMPL::pick_up(v1, v2);                  // { 0, 2, 4, 6, 8, a, c, e }
  vs = SORT_IMPL::pick_down(v1, v2);                // { 1, 3, 5, 7, 9, b, d, f }
  /** restore order */
  v1 = _mm256_unpacklo_ps(vl, vs);                  // { 0, 1, 2, 3, 8, 9, a, b }
  v2 = _mm256_unpackhi_ps(vl, vs);                  // { 4, 5, 6, 7, c, d, e, f }
  up = _mm256_permute2f128_ps(v1, v2, 0x20);        // { 0, 1, 2, 3, 4, 5, 6, 7 }
  down = _mm256_permute2f128_ps(v1, v2, 0x31);      // { 8, 9, a, b, c, d, e, f }
}

#if __AVX512F__
template <typename SORT_IMPL>
void sort_32_f32_avx_bitonic_input(__m512 &up, __m512 &down) {
  /** step 1 */
  __m512 vl = SORT_IMPL::pick_up(up, down);       // { 00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15 }
  __m512 vs = SORT_IMPL::pick_down(up, down);     // { 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 }
  /** step 2 */
  __m512 v1 = _mm512_shuffle_f32x4(vl, vs, 0x44); // { 00, 01, 02, 03, 04, 05, 06, 07, 16, 17, 18, 19, 20, 21, 22, 23 }
  __m512 v2 = _mm512_shuffle_f32x4(vl, vs, 0xee); // { 08, 09, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31 }
  vl = SORT_IMPL::pick_up(v1, v2);                // { 00, 01, 02, 03, 04, 05, 06, 07, 16, 17, 18, 19, 20, 21, 22, 23 }
  vs = SORT_IMPL::pick_down(v1, v2);              // { 08, 09, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31 }
  /** step 3 */
  v1 = _mm512_shuffle_f32x4(vl, vs, 0x88);        // { 00, 01, 02, 03, 16, 17, 18, 19, 08, 09, 10, 11, 24, 25, 26, 27 }
  v2 = _mm512_shuffle_f32x4(vl, vs, 0xdd);        // { 04, 05, 06, 07, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31 }
  vl = SORT_IMPL::pick_up(v1, v2);                // { 00, 01, 02, 03, 16, 17, 18, 19, 08, 09, 10, 11, 24, 25, 26, 27 }
  vs = SORT_IMPL::pick_down(v1, v2);              // { 04, 05, 06, 07, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31 }
  /** step 4 */
  v1 = _mm512_unpacklo_ps(vl, vs);                // { 00, 04, 01, 05, 16, 20, 17, 21, 08, 12, 09, 13, 24, 28, 25, 29 }
  v2 = _mm512_unpackhi_ps(vl, vs);                // { 02, 06, 03, 07, 18, 22, 19, 23, 10, 14, 11, 15, 26, 30, 27, 31 }
  vl = SORT_IMPL::pick_up(v1, v2);                // { 00, 04, 01, 05, 16, 20, 17, 21, 08, 12, 09, 13, 24, 28, 25, 29 }
  vs = SORT_IMPL::pick_down(v1, v2);              // { 02, 06, 03, 07, 18, 22, 19, 23, 10, 14, 11, 15, 26, 30, 27, 31 }
  /** step 5 */
  v1 = _mm512_unpacklo_ps(vl, vs);                // { 00, 02, 04, 06, 16, 18, 20, 22, 08, 10, 12, 14, 24, 26, 28, 30 }
  v2 = _mm512_unpackhi_ps(vl, vs);                // { 01, 03, 05, 07, 17, 19, 21, 23, 09, 11, 13, 15, 25, 27, 29, 31 }
  vl = SORT_IMPL::pick_up(v1, v2);                // { 00, 02, 04, 06, 16, 18, 20, 22, 08, 10, 12, 14, 24, 26, 28, 30 }
  vs = SORT_IMPL::pick_down(v1, v2);              // { 01, 03, 05, 07, 17, 19, 21, 23, 09, 11, 13, 15, 25, 27, 29, 31 }
  /** reorder data */
  const __m512i perm_idx_l = _mm512_setr_epi32(
    0x00, 0x10, 0x01, 0x11, 0x02, 0x12, 0x03, 0x13,
    0x08, 0x18, 0x09, 0x19, 0x0a, 0x1a, 0x0b, 0x1b);
  const __m512i perm_idx_h = _mm512_setr_epi32(
    0x04, 0x14, 0x05, 0x15, 0x06, 0x16, 0x07, 0x17,
    0x0c, 0x1c, 0x0d, 0x1d, 0x0e, 0x1e, 0x0f, 0x1f);
  up = _mm512_permutex2var_ps(vl, perm_idx_l, vs); // { 00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15 }
  down = _mm512_permutex2var_ps(vl, perm_idx_h, vs); // { 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 }
}
#else
template <typename SORT_IMPL>
void sort_32_f32_avx_bitonic_input(__m256 &v0, __m256 &v1,
  __m256 &v2, __m256 &v3) {
  __m256 n0 = SORT_IMPL::pick_up(v0, v2), n1 = SORT_IMPL::pick_up(v1, v3);
  v2 = SORT_IMPL::pick_down(v0, v2);
  v3 = SORT_IMPL::pick_down(v1, v3);
  sort_16_f32_avx_bitonic_input<SORT_IMPL>(n0, n1);
  sort_16_f32_avx_bitonic_input<SORT_IMPL>(v2, v3);
  v0 = n0;
  v1 = n1;
}
#endif

#if __AVX512F__
template <typename SORT_IMPL>
void sort_32_f32_avx(__m512 &up, __m512 &down) {
  /** stage 1 */
  __m512 v1 = _mm512_shuffle_ps(up, down, 0xcc); // { 00, 03, 16, 19, 04, 07, 20, 23, 08, 11, 24, 27, 12, 15, 28, 31 }
  __m512 v2 = _mm512_shuffle_ps(up, down, 0x99); // { 01, 02, 17, 18, 05, 06, 21, 22, 09, 10, 25, 26, 13, 14, 29, 30 }
  __m512 vl = SORT_IMPL::pick_up(v1, v2);        // { 00, 03, 16, 19, 04, 07, 20, 23, 08, 11, 24, 27, 12, 15, 28, 31 }
  __m512 vs = SORT_IMPL::pick_down(v1, v2);      // { 01, 02, 17, 18, 05, 06, 21, 22, 09, 10, 25, 26, 13, 14, 29, 30 }
  /** stage 2 */
  v2 = _mm512_permute_ps(vs, 0xb1);              // { 02, 01, 18, 17, 06, 05, 22, 21, 10, 09, 26, 25, 14, 13, 30, 29 }
  vs = SORT_IMPL::pick_down(vl, v2);             // { 02, 03, 18, 19, 04, 05, 20, 21, 10, 11, 26, 27, 12, 13, 28, 29 }
  vl = SORT_IMPL::pick_up(vl, v2);               // { 00, 01, 16, 17, 06, 07, 22, 23, 08, 09, 24, 25, 14, 15, 30, 31 }
  v1 = _mm512_shuffle_ps(vl, vs, 0x88);          // { 00, 16, 02, 18, 06, 22, 04, 20, 08, 24, 10, 26, 14, 30, 12, 28 }
  v2 = _mm512_shuffle_ps(vl, vs, 0xdd);          // { 01, 17, 03, 19, 07, 23, 05, 21, 09, 25, 11, 27, 15, 31, 13, 29 }
  vl = SORT_IMPL::pick_up(v1, v2);               // { 00, 16, 02, 18, 07, 23, 05, 21, 08, 24, 10, 26, 15, 31, 13, 29 }
  vs = SORT_IMPL::pick_down(v1, v2);             // { 01, 17, 03, 19, 06, 22, 04, 20, 09, 25, 11, 27, 14, 30, 12, 28 }
  /** stage 3 */
  v1 = _mm512_permute_ps(vl, 0x4e);              // { 02, 18, 00, 16, 05, 21, 07, 23, 10, 26, 08, 24, 13, 29, 15, 31 }
  v2 = _mm512_shuffle_f32x4(vs, vs, 0xb1);       // { 06, 22, 04, 20, 01, 17, 03, 19, 14, 30, 12, 28, 09, 25, 11, 27 }
  vl = SORT_IMPL::pick_up(v1, v2);               // { 02, 18, 00, 16, 01, 17, 03, 19, 14, 30, 12, 28, 13, 29, 15, 31 }
  vs = SORT_IMPL::pick_down(v1, v2);             // { 06, 22, 04, 20, 05, 21, 07, 23, 10, 26, 08, 24, 09, 25, 11, 27 }
  v1 = _mm512_shuffle_ps(vl, vs, 0x44);          // { 02, 18, 06, 22, 01, 17, 05, 21, 14, 30, 10, 26, 13, 29, 09, 25 }
  v2 = _mm512_shuffle_ps(vl, vs, 0xee);          // { 00, 16, 04, 20, 03, 19, 07, 23, 12, 28, 08, 24, 15, 31, 11, 27 }
  vl = SORT_IMPL::pick_up(v1, v2);               // { 00, 16, 04, 20, 01, 17, 05, 21, 14, 30, 10, 26, 15, 31, 11, 27 }
  vs = SORT_IMPL::pick_down(v1, v2);             // { 02, 18, 06, 22, 03, 19, 07, 23, 12, 28, 08, 24, 13, 29, 09, 25 }
  v1 = _mm512_shuffle_f32x4(vl, vs, 0x88);       // { 00, 16, 04, 20, 14, 30, 10, 26, 02, 18, 06, 22, 12, 28, 08, 24 }
  v2 = _mm512_shuffle_f32x4(vl, vs, 0xdd);       // { 01, 17, 05, 21, 15, 31, 11, 27, 03, 19, 07, 23, 13, 29, 09, 25 }
  vl = SORT_IMPL::pick_up(v1, v2);               // { 00, 16, 04, 20, 15, 31, 11, 27, 02, 18, 06, 22, 13, 29, 09, 25 }
  vs = SORT_IMPL::pick_down(v1, v2);             // { 01, 17, 05, 21, 14, 30, 10, 26, 03, 19, 07, 23, 12, 28, 08, 24 }
  /** stage 4 */
  v1 = _mm512_permute_ps(vl, 0x4e);              // { 04, 20, 00, 16, 11, 27, 15, 31, 06, 22, 02, 18, 09, 25, 13, 29 }
  v2 = _mm512_shuffle_f32x4(vs, vs, 0x1b);       // { 12, 28, 08, 24, 03, 19, 07, 23, 14, 30, 10, 26, 01, 17, 05, 21 }
  vl = SORT_IMPL::pick_up(v1, v2);               // { 04, 28, 00, 24, 03, 27, 07, 31, 06, 30, 02, 26, 01, 25, 05, 29 }
  vs = SORT_IMPL::pick_down(v1, v2);             // { 12, 20, 08, 16, 11, 19, 15, 23, 14, 22, 10, 18, 09, 17, 13, 21 }
  v1 = _mm512_shuffle_ps(vl, vs, 0x44);          // { 04, 28, 12, 20, 03, 27, 11, 19, 06, 30, 14, 22, 01, 25, 09, 17 }
  v2 = _mm512_shuffle_ps(vl, vs, 0xee);          // { 00, 24, 08, 16, 07, 31, 15, 23, 02, 26, 10, 18, 05, 29, 13, 21 }
  vl = SORT_IMPL::pick_up(v1, v2);               // { 00, 28, 08, 20, 03, 31, 11, 23, 02, 30, 10, 22, 01, 29, 09, 21 }
  vs = SORT_IMPL::pick_down(v1, v2);             // { 04, 24, 12, 16, 07, 27, 15, 19, 06, 26, 14, 18, 05, 25, 13, 17 }
  v1 = _mm512_shuffle_f32x4(vl, vs, 0x44);       // { 00, 28, 08, 20, 03, 31, 11, 23, 04, 24, 12, 16, 07, 27, 15, 19 }
  v2 = _mm512_shuffle_f32x4(vl, vs, 0xee);       // { 02, 30, 10, 22, 01, 29, 09, 21, 06, 26, 14, 18, 05, 25, 13, 17 }
  vl = SORT_IMPL::pick_up(v1, v2);               // { 00, 30, 08, 22, 01, 31, 09, 23, 04, 26, 12, 18, 05, 27, 13, 19 }
  vs = SORT_IMPL::pick_down(v1, v2);             // { 02, 28, 10, 20, 03, 29, 11, 21, 06, 24, 14, 16, 07, 25, 15, 17 }
  v1 = _mm512_shuffle_f32x4(vl, vs, 0x88);       // { 00, 30, 08, 22, 04, 26, 12, 18, 02, 28, 10, 20, 06, 24, 14, 16 }
  v2 = _mm512_shuffle_f32x4(vl, vs, 0xdd);       // { 01, 31, 09, 23, 05, 27, 13, 19, 03, 29, 11, 21, 07, 25, 15, 17 }
  vl = SORT_IMPL::pick_up(v1, v2);               // { 00, 31, 08, 23, 04, 27, 12, 19, 02, 29, 10, 21, 06, 25, 14, 17 }
  vs = SORT_IMPL::pick_down(v1, v2);             // { 01, 30, 09, 22, 05, 26, 13, 18, 03, 28, 11, 20, 07, 24, 15, 16 }
  /** stage 5 */
  v1 = _mm512_permute_ps(vl, 0x1b);              // { 23, 08, 31, 00, 19, 12, 27, 04, 21, 10, 29, 02, 17, 14, 25, 06 }
  v2 = _mm512_shuffle_f32x4(vs, vs, 0x1b);       // { 07, 24, 15, 16, 03, 28, 11, 20, 05, 26, 13, 18, 01, 30, 09, 22 }
  vl = SORT_IMPL::pick_up(v1, v2);               // { 07, 08, 15, 00, 03, 12, 11, 04, 05, 10, 13, 02, 01, 14, 09, 06 }
  vs = SORT_IMPL::pick_down(v1, v2);             // { 23, 24, 31, 16, 19, 28, 27, 20, 21, 26, 29, 18, 17, 30, 25, 22 }
  v1 = _mm512_unpacklo_ps(vl, vs);               // { 07, 23, 08, 24, 03, 19, 12, 28, 05, 21, 10, 26, 01, 17, 14, 30 }
  v2 = _mm512_unpackhi_ps(vl, vs);               // { 15, 31, 00, 16, 11, 27, 04, 20, 13, 29, 02, 18, 09, 25, 06, 22 }
  vl = SORT_IMPL::pick_up(v1, v2);               // { 07, 23, 00, 16, 03, 19, 04, 20, 05, 21, 02, 18, 01, 17, 06, 22 }
  vs = SORT_IMPL::pick_down(v1, v2);             // { 15, 31, 08, 24, 11, 27, 12, 28, 13, 29, 10, 26, 09, 25, 14, 30 }
  v1 = _mm512_shuffle_f32x4(vl, vs, 0x88);       // { 07, 23, 00, 16, 05, 21, 02, 18, 15, 31, 08, 24, 13, 29, 10, 26 }
  v2 = _mm512_shuffle_f32x4(vl, vs, 0xdd);       // { 03, 19, 04, 20, 01, 17, 06, 22, 11, 27, 12, 28, 09, 25, 14, 30 }
  vl = SORT_IMPL::pick_up(v1, v2);               // { 03, 19, 00, 16, 01, 17, 02, 18, 11, 27, 08, 24, 09, 25, 10, 26 }
  vs = SORT_IMPL::pick_down(v1, v2);             // { 07, 23, 04, 20, 05, 21, 06, 22, 15, 31, 12, 28, 13, 29, 14, 30 }
  v1 = _mm512_shuffle_f32x4(vl, vs, 0x88);       // { 03, 19, 00, 16, 11, 27, 08, 24, 07, 23, 04, 20, 15, 31, 12, 28 }
  v2 = _mm512_shuffle_f32x4(vl, vs, 0xdd);       // { 01, 17, 02, 18, 09, 25, 10, 26, 05, 21, 06, 22, 13, 29, 14, 30 }
  vl = SORT_IMPL::pick_up(v1, v2);               // { 01, 17, 00, 16, 09, 25, 08, 24, 05, 21, 04, 20, 13, 29, 12, 28 }
  vs = SORT_IMPL::pick_down(v1, v2);             // { 03, 19, 02, 18, 11, 27, 10, 26, 07, 23, 06, 22, 15, 31, 14, 30 }
  v1 = _mm512_unpacklo_ps(vl, vs);               // { 01, 03, 17, 19, 09, 11, 25, 27, 05, 07, 21, 23, 13, 15, 29, 31 }
  v2 = _mm512_unpackhi_ps(vl, vs);               // { 00, 02, 16, 18, 08, 10, 24, 26, 04, 06, 20, 22, 12, 14, 28, 30 }
  vl = SORT_IMPL::pick_up(v1, v2);               // { 00, 02, 16, 18, 08, 10, 24, 26, 04, 06, 20, 22, 12, 14, 28, 30 }
  vs = SORT_IMPL::pick_down(v1, v2);             // { 01, 03, 17, 19, 09, 11, 25, 27, 05, 07, 21, 23, 13, 15, 29, 31 }
  /** restore order */
  v1 = _mm512_unpacklo_ps(vl, vs);               // { 00, 01, 02, 03, 08, 09, 10, 11, 04, 05, 06, 07, 12, 13, 14, 15 }
  v2 = _mm512_unpackhi_ps(vl, vs);               // { 16, 17, 18, 19, 24, 25, 26, 27, 20, 21, 22, 23, 28, 29, 30, 31 }
  up = _mm512_shuffle_f32x4(v1, v1, 0xd8);       // { 00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15 }
  down = _mm512_shuffle_f32x4(v2, v2, 0xd8);     // { 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 }
}
#else
template <typename SORT_IMPL>
void sort_32_f32_avx(__m256 &v0, __m256 &v1, __m256 &v2, __m256 &v3) {
  sort_16_f32_avx<SORT_IMPL>(v0, v1);
  sort_16_f32_avx<typename SORT_IMPL::reverse_type>(v2, v3);
  sort_32_f32_avx_bitonic_input<SORT_IMPL>(v0, v1, v2, v3);
}
#endif

#if __AVX512F__
template <typename SORT_IMPL>
void sort_64_f32_avx_bitonic_input(__m512 &v0, __m512 &v1,
  __m512 &v2, __m512 &v3) {

  __m512 n0 = SORT_IMPL::pick_up(v0, v2), n1 = SORT_IMPL::pick_up(v1, v3);
  v2 = SORT_IMPL::pick_down(v0, v2);
  v3 = SORT_IMPL::pick_down(v1, v3);
  sort_32_f32_avx_bitonic_input<SORT_IMPL>(n0, n1);
  sort_32_f32_avx_bitonic_input<SORT_IMPL>(v2, v3);
  v0 = n0;
  v1 = n1;
}
#else
template <typename SORT_IMPL>
void sort_64_f32_avx_bitonic_input(float *dat) {
  __m256 v0 = _mm256_loadu_ps(dat), v1 = _mm256_loadu_ps(dat + 8),
    v2 = _mm256_loadu_ps(dat + 16), v3 = _mm256_loadu_ps(dat + 24);
  __m256 v4 = _mm256_loadu_ps(dat + 32), v5 = _mm256_loadu_ps(dat + 40),
    v6 = _mm256_loadu_ps(dat + 48), v7 = _mm256_loadu_ps(dat + 56);
  __m256 n0 = SORT_IMPL::pick_up(v0, v4), n1 = SORT_IMPL::pick_up(v1, v5),
    n2 = SORT_IMPL::pick_up(v2, v6), n3 = SORT_IMPL::pick_up(v3, v7);
  v4 = SORT_IMPL::pick_down(v0, v4);
  v5 = SORT_IMPL::pick_down(v1, v5);
  v6 = SORT_IMPL::pick_down(v2, v6);
  v7 = SORT_IMPL::pick_down(v3, v7);
  sort_32_f32_avx_bitonic_input<SORT_IMPL>(n0, n1, n2, n3);
  _mm256_storeu_ps(dat, n0);
  _mm256_storeu_ps(dat + 8, n1);
  _mm256_storeu_ps(dat + 16, n2);
  _mm256_storeu_ps(dat + 24, n3);
  sort_32_f32_avx_bitonic_input<SORT_IMPL>(v4, v5, v6, v7);
  _mm256_storeu_ps(dat + 32, v4);
  _mm256_storeu_ps(dat + 40, v5);
  _mm256_storeu_ps(dat + 48, v6);
  _mm256_storeu_ps(dat + 56, v7);
}
#endif

#if __AVX512F__
template <typename SORT_IMPL>
void sort_64_f32_avx(__m512 &v0, __m512 &v1, __m512 &v2, __m512 &v3) {
  sort_32_f32_avx<SORT_IMPL>(v0, v1);
  sort_32_f32_avx<typename SORT_IMPL::reverse_type>(v2, v3);
  sort_64_f32_avx_bitonic_input<SORT_IMPL>(v0, v1, v2, v3);
}
#else
template <typename SORT_IMPL>
void sort_64_f32_avx(float *dat) {
  __m256 v0 = _mm256_loadu_ps(dat), v1 = _mm256_loadu_ps(dat + 8),
    v2 = _mm256_loadu_ps(dat + 16), v3 = _mm256_loadu_ps(dat + 24);
  sort_32_f32_avx<SORT_IMPL>(v0, v1, v2, v3);
  _mm256_storeu_ps(dat, v0);
  _mm256_storeu_ps(dat + 8, v1);
  _mm256_storeu_ps(dat + 16, v2);
  _mm256_storeu_ps(dat + 24, v3);
  __m256 v4 = _mm256_loadu_ps(dat + 32), v5 = _mm256_loadu_ps(dat + 40),
    v6 = _mm256_loadu_ps(dat + 48), v7 = _mm256_loadu_ps(dat + 56);
  sort_32_f32_avx<typename SORT_IMPL::reverse_type>(v4, v5, v6, v7);
  _mm256_storeu_ps(dat + 32, v4);
  _mm256_storeu_ps(dat + 40, v5);
  _mm256_storeu_ps(dat + 48, v6);
  _mm256_storeu_ps(dat + 56, v7);
  sort_64_f32_avx_bitonic_input<SORT_IMPL>(dat);
}
#endif

template <typename SORT_IMPL>
void part_32v32_f32(float *up, float *down) {
#if __AVX512F__
  __m512 v0 = _mm512_loadu_ps(up), v1 = _mm512_loadu_ps(up + 16);
  __m512 v2 = _mm512_loadu_ps(down), v3 = _mm512_loadu_ps(down + 16);
  _mm512_storeu_ps(up, SORT_IMPL::pick_up(v0, v2));
  _mm512_storeu_ps(up + 16, SORT_IMPL::pick_up(v1, v3));
  _mm512_storeu_ps(down, SORT_IMPL::pick_down(v0, v2));
  _mm512_storeu_ps(down + 16, SORT_IMPL::pick_down(v1, v3));
#else
  __m256 v0 = _mm256_loadu_ps(up), v1 = _mm256_loadu_ps(up + 8),
    v2 = _mm256_loadu_ps(up + 16), v3 = _mm256_loadu_ps(up + 24);
  __m256 v4 = _mm256_loadu_ps(down), v5 = _mm256_loadu_ps(down + 8),
    v6 = _mm256_loadu_ps(down + 16), v7 = _mm256_loadu_ps(down + 24);
  _mm256_storeu_ps(up, SORT_IMPL::pick_up(v0, v4));
  _mm256_storeu_ps(up + 8, SORT_IMPL::pick_up(v1, v5));
  _mm256_storeu_ps(up + 16, SORT_IMPL::pick_up(v2, v6));
  _mm256_storeu_ps(up + 24, SORT_IMPL::pick_up(v3, v7));
  _mm256_storeu_ps(down, SORT_IMPL::pick_down(v0, v4));
  _mm256_storeu_ps(down + 8, SORT_IMPL::pick_down(v1, v5));
  _mm256_storeu_ps(down + 16, SORT_IMPL::pick_down(v2, v6));
  _mm256_storeu_ps(down + 24, SORT_IMPL::pick_down(v3, v7));
#endif
}

#if __AVX512F__
template <typename SORT_IMPL>
void sort_128_f32_avx_bitonic_input(float *dat) {
  __m512 v0 = _mm512_loadu_ps(dat), v1 = _mm512_loadu_ps(dat + 16),
    v2 = _mm512_loadu_ps(dat + 32), v3 = _mm512_loadu_ps(dat + 48);
  __m512 v4 = _mm512_loadu_ps(dat + 64), v5 = _mm512_loadu_ps(dat + 80),
    v6 = _mm512_loadu_ps(dat + 96), v7 = _mm512_loadu_ps(dat + 112);
  __m512 n0 = SORT_IMPL::pick_up(v0, v4), n1 = SORT_IMPL::pick_up(v1, v5),
    n2 = SORT_IMPL::pick_up(v2, v6), n3 = SORT_IMPL::pick_up(v3, v7);
  v4 = SORT_IMPL::pick_down(v0, v4);
  v5 = SORT_IMPL::pick_down(v1, v5);
  v6 = SORT_IMPL::pick_down(v2, v6);
  v7 = SORT_IMPL::pick_down(v3, v7);
  sort_64_f32_avx_bitonic_input<SORT_IMPL>(n0, n1, n2, n3);
  _mm512_storeu_ps(dat, n0);
  _mm512_storeu_ps(dat + 16, n1);
  _mm512_storeu_ps(dat + 32, n2);
  _mm512_storeu_ps(dat + 48, n3);
  sort_64_f32_avx_bitonic_input<SORT_IMPL>(v4, v5, v6, v7);
  _mm512_storeu_ps(dat + 64, v4);
  _mm512_storeu_ps(dat + 80, v5);
  _mm512_storeu_ps(dat + 96, v6);
  _mm512_storeu_ps(dat + 112, v7);
}
#else
template <typename SORT_IMPL>
void sort_128_f32_avx_bitonic_input(float *dat) {
  part_32v32_f32<SORT_IMPL>(dat, dat + 64);
  part_32v32_f32<SORT_IMPL>(dat + 32, dat + 96);
  sort_64_f32_avx_bitonic_input<SORT_IMPL>(dat);
  sort_64_f32_avx_bitonic_input<SORT_IMPL>(dat + 64);
}
#endif

#if __AVX512F__
template <typename SORT_IMPL>
void sort_128_f32_avx(float *dat) {
  __m512 v0 = _mm512_loadu_ps(dat), v1 = _mm512_loadu_ps(dat + 16),
    v2 = _mm512_loadu_ps(dat + 32), v3 = _mm512_loadu_ps(dat + 48);
  sort_64_f32_avx<SORT_IMPL>(v0, v1, v2, v3);
  _mm512_storeu_ps(dat, v0);
  _mm512_storeu_ps(dat + 16, v1);
  _mm512_storeu_ps(dat + 32, v2);
  _mm512_storeu_ps(dat + 48, v3);
  __m512 v4 = _mm512_loadu_ps(dat + 64), v5 = _mm512_loadu_ps(dat + 80),
    v6 = _mm512_loadu_ps(dat + 96), v7 = _mm512_loadu_ps(dat + 112);
  sort_64_f32_avx<typename SORT_IMPL::reverse_type>(v4, v5, v6, v7);
  _mm512_storeu_ps(dat + 64, v4);
  _mm512_storeu_ps(dat + 80, v5);
  _mm512_storeu_ps(dat + 96, v6);
  _mm512_storeu_ps(dat + 112, v7);
  sort_128_f32_avx_bitonic_input<SORT_IMPL>(dat);
}
#else
template <typename SORT_IMPL>
void sort_128_f32_avx(float *dat) {
  sort_64_f32_avx<SORT_IMPL>(dat);
  sort_64_f32_avx<typename SORT_IMPL::reverse_type>(dat + 64);
  sort_128_f32_avx_bitonic_input<SORT_IMPL>(dat);
}
#endif

template <typename SORT_IMPL>
void top_16_f32_avx(float *dat, std::size_t folds) {
  if (!folds) return;
  __m256 v0 = _mm256_loadu_ps(dat), v1 = _mm256_loadu_ps(dat + 8);
  sort_16_f32_avx<SORT_IMPL>(v0, v1);
  float * const begin_ = dat;
  for (; folds > 1; folds--) {
    __m256 v2 = _mm256_loadu_ps(dat + 16), v3 = _mm256_loadu_ps(dat + 24);
    sort_16_f32_avx<typename SORT_IMPL::reverse_type>(v2, v3);
    _mm256_storeu_ps(dat + 16, SORT_IMPL::pick_down(v0, v2));
    _mm256_storeu_ps(dat + 24, SORT_IMPL::pick_down(v1, v3));
    v0 = SORT_IMPL::pick_up(v0, v2);
    v1 = SORT_IMPL::pick_up(v1, v3);
    sort_16_f32_avx_bitonic_input<SORT_IMPL>(v0, v1);
    dat += 16;
  }
  _mm256_storeu_ps(begin_, v0);
  _mm256_storeu_ps(begin_ + 8, v1);
}

#if __AVX512F__
template <typename SORT_IMPL>
void top_32_f32_avx(float *dat, std::size_t folds) {
  if (!folds) return;
  __m512 v0 = _mm512_loadu_ps(dat), v1 = _mm512_loadu_ps(dat + 16);
  sort_32_f32_avx<SORT_IMPL>(v0, v1);
  float * const begin_ = dat;
  for (; folds > 1; folds--) {
    dat += 32;
    __m512 v2 = _mm512_loadu_ps(dat), v3 = _mm512_loadu_ps(dat + 16);
    sort_32_f32_avx<typename SORT_IMPL::reverse_type>(v2, v3);
    _mm512_storeu_ps(dat, SORT_IMPL::pick_down(v0, v2));
    _mm512_storeu_ps(dat + 16, SORT_IMPL::pick_down(v1, v3));
    v0 = SORT_IMPL::pick_up(v0, v2);
    v1 = SORT_IMPL::pick_up(v1, v3);
    sort_32_f32_avx_bitonic_input<SORT_IMPL>(v0, v1);
  }
  _mm512_storeu_ps(begin_, v0);
  _mm512_storeu_ps(begin_ + 16, v1);
}
#else
template <typename SORT_IMPL>
void top_32_f32_avx(float *dat, std::size_t folds) {
  if (!folds) return;
  __m256 v0 = _mm256_loadu_ps(dat), v1 = _mm256_loadu_ps(dat + 8),
    v2 = _mm256_loadu_ps(dat + 16), v3 = _mm256_loadu_ps(dat + 24);
  sort_32_f32_avx<SORT_IMPL>(v0, v1, v2, v3);
  float * const begin_ = dat;
  for (; folds > 1; folds--) {
    __m256 v4 = _mm256_loadu_ps(dat + 32), v5 = _mm256_loadu_ps(dat + 40),
      v6 = _mm256_loadu_ps(dat + 48), v7 = _mm256_loadu_ps(dat + 56);
    sort_32_f32_avx<typename SORT_IMPL::reverse_type>(v4, v5, v6, v7);
    _mm256_storeu_ps(dat + 32, SORT_IMPL::pick_down(v0, v4));
    _mm256_storeu_ps(dat + 40, SORT_IMPL::pick_down(v1, v5));
    _mm256_storeu_ps(dat + 48, SORT_IMPL::pick_down(v2, v6));
    _mm256_storeu_ps(dat + 56, SORT_IMPL::pick_down(v3, v7));
    v0 = SORT_IMPL::pick_up(v0, v4);
    v1 = SORT_IMPL::pick_up(v1, v5);
    v2 = SORT_IMPL::pick_up(v2, v6);
    v3 = SORT_IMPL::pick_up(v3, v7);
    sort_32_f32_avx_bitonic_input<SORT_IMPL>(v0, v1, v2, v3);
    dat += 32;
  }
  _mm256_storeu_ps(begin_, v0);
  _mm256_storeu_ps(begin_ + 8, v1);
  _mm256_storeu_ps(begin_ + 16, v2);
  _mm256_storeu_ps(begin_ + 24, v3);
}
#endif

#if __AVX512F__
template <typename SORT_IMPL>
void top_64_f32_avx(float *dat, std::size_t folds) {
  if (!folds) return;
  __m512 v0 = _mm512_loadu_ps(dat), v1 = _mm512_loadu_ps(dat + 16),
    v2 = _mm512_loadu_ps(dat + 32), v3 = _mm512_loadu_ps(dat + 48);
  sort_64_f32_avx<SORT_IMPL>(v0, v1, v2, v3);
  float * const begin_ = dat;
  for (; folds > 1; folds--) {
    dat += 64;
    __m512 v4 = _mm512_loadu_ps(dat), v5 = _mm512_loadu_ps(dat + 16),
      v6 = _mm512_loadu_ps(dat + 32), v7 = _mm512_loadu_ps(dat + 48);
    sort_64_f32_avx<typename SORT_IMPL::reverse_type>(v4, v5, v6, v7);
    _mm512_storeu_ps(dat, SORT_IMPL::pick_down(v0, v4));
    _mm512_storeu_ps(dat + 16, SORT_IMPL::pick_down(v1, v5));
    _mm512_storeu_ps(dat + 32, SORT_IMPL::pick_down(v2, v6));
    _mm512_storeu_ps(dat + 48, SORT_IMPL::pick_down(v3, v7));
    v0 = SORT_IMPL::pick_up(v0, v4);
    v1 = SORT_IMPL::pick_up(v1, v5);
    v2 = SORT_IMPL::pick_up(v2, v6);
    v3 = SORT_IMPL::pick_up(v3, v7);
    sort_64_f32_avx_bitonic_input<SORT_IMPL>(v0, v1, v2, v3);
  }
  _mm512_storeu_ps(begin_, v0);
  _mm512_storeu_ps(begin_ + 16, v1);
  _mm512_storeu_ps(begin_ + 32, v2);
  _mm512_storeu_ps(begin_ + 48, v3);
}
#else
template <typename SORT_IMPL>
void top_64_f32_avx(float *dat, std::size_t folds) {
  if (!folds) return;
  sort_64_f32_avx<SORT_IMPL>(dat);
  float * const begin_ = dat;
  for (; folds > 1; folds--) {
    dat += 64;
    sort_64_f32_avx<typename SORT_IMPL::reverse_type>(dat);
    part_32v32_f32<SORT_IMPL>(begin_, dat);
    part_32v32_f32<SORT_IMPL>(begin_ + 32, dat + 32);
    sort_64_f32_avx_bitonic_input<SORT_IMPL>(begin_);
  }
}
#endif

template <typename SORT_IMPL>
void top_128_f32_avx(float *dat, std::size_t folds) {
  if (!folds) return;
  sort_128_f32_avx<SORT_IMPL>(dat);
  float * const begin_ = dat;
  for (; folds > 1; folds--) {
    dat += 128;
    sort_128_f32_avx<typename SORT_IMPL::reverse_type>(dat);
    part_32v32_f32<SORT_IMPL>(begin_, dat);
    part_32v32_f32<SORT_IMPL>(begin_ + 32, dat + 32);
    part_32v32_f32<SORT_IMPL>(begin_ + 64, dat + 64);
    part_32v32_f32<SORT_IMPL>(begin_ + 96, dat + 96);
    sort_128_f32_avx_bitonic_input<SORT_IMPL>(begin_);
  }
}

struct SORT_DECR;

struct SORT_INCR {
  typedef SORT_DECR reverse_type;
  static __m256 pick_up(__m256 in1, __m256 in2) {
    return _mm256_min_ps(in1, in2);
  }
  static __m256 pick_down(__m256 in1, __m256 in2) {
    return _mm256_max_ps(in1, in2);
  }
};

struct SORT_DECR {
  typedef SORT_INCR reverse_type;
  static __m256 pick_up(__m256 in1, __m256 in2) {
    return _mm256_max_ps(in1, in2);
  }
  static __m256 pick_down(__m256 in1, __m256 in2) {
    return _mm256_min_ps(in1, in2);
  }
};

#if __AVX512F__
struct SORT_DECR_512;

struct SORT_INCR_512 {
  typedef SORT_DECR_512 reverse_type;
  static __m512 pick_up(__m512 in1, __m512 in2) {
    return _mm512_min_ps(in1, in2);
  }
  static __m512 pick_down(__m512 in1, __m512 in2) {
    return _mm512_max_ps(in1, in2);
  }
};

struct SORT_DECR_512 {
  typedef SORT_INCR_512 reverse_type;
  static __m512 pick_up(__m512 in1, __m512 in2) {
    return _mm512_max_ps(in1, in2);
  }
  static __m512 pick_down(__m512 in1, __m512 in2) {
    return _mm512_min_ps(in1, in2);
  }
};
#endif
#endif

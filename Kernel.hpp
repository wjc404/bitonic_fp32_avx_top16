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

template <typename SORT_IMPL>
void sort_32_f32_avx(__m256 &v0, __m256 &v1, __m256 &v2, __m256 &v3) {
  sort_16_f32_avx<SORT_IMPL>(v0, v1);
  sort_16_f32_avx<typename SORT_IMPL::reverse_type>(v2, v3);
  sort_32_f32_avx_bitonic_input<SORT_IMPL>(v0, v1, v2, v3);
}

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

template <typename SORT_IMPL>
void part_32v32_f32(float *up, float *down) {
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
}

template <typename SORT_IMPL>
void sort_128_f32_avx_bitonic_input(float *dat) {
  part_32v32_f32<SORT_IMPL>(dat, dat + 64);
  part_32v32_f32<SORT_IMPL>(dat + 32, dat + 96);
  sort_64_f32_avx_bitonic_input<SORT_IMPL>(dat);
  sort_64_f32_avx_bitonic_input<SORT_IMPL>(dat + 64);
}

template <typename SORT_IMPL>
void sort_128_f32_avx(float *dat) {
  sort_64_f32_avx<SORT_IMPL>(dat);
  sort_64_f32_avx<typename SORT_IMPL::reverse_type>(dat + 64);
  sort_128_f32_avx_bitonic_input<SORT_IMPL>(dat);
}

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

#endif

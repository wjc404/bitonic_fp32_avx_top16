#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>
#include <cstdio>
#include <iomanip>

#include "Kernel.hpp"

using std::chrono::steady_clock;

int test_bitonic_topk_fp32(void (*func)(float*, std::size_t),
  const std::size_t k, const std::size_t folds, std::size_t loops) {

  if (k < 2 || folds < 2) return -1;
  std::cout << "test with k = " << k << " and folds = " << folds << std::endl;
  std::cout << "    nth_element(sec)    bitonic_avx(sec)\n";
  const std::size_t size = k * folds;
  std::vector<float> ref(size), tst(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1, 1);
  for (; loops; loops--) {
    for (std::size_t i = 0; i < size; ++i) {
      ref[i] = tst[i] = dist(gen);
    }
    auto st = steady_clock::now();
    std::nth_element(ref.begin(), ref.begin() + k - 1, ref.end());
    auto et = steady_clock::now();
    std::cout << std::scientific << std::setw(20)
      << std::chrono::duration<double>(et - st).count();
    std::sort(ref.begin(), ref.begin() + k - 1);
    std::sort(ref.begin() + k, ref.end());

    st = steady_clock::now();
    (*func)(tst.data(), folds);
    et = steady_clock::now();
    std::cout << std::setw(20) << std::chrono::duration<double>(et - st).count()
      << std::endl;
    std::sort(tst.begin() + k, tst.end());

    for (std::size_t i = 0; i < size; ++i) {
      if (ref[i] != tst[i]) {
        std::cout << "inconsistency at index = " << i << " for k = "
          << k << " and folds = " << folds << std::endl;
        return -1;
      }
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  const std::size_t folds = argc > 1 ? std::atoi(argv[1]) : 300;
  constexpr unsigned int iters = 6;
  test_bitonic_topk_fp32(top_16_f32_avx<SORT_INCR>, 16, folds * 8, iters);
  test_bitonic_topk_fp32(top_32_f32_avx<SORT_INCR>, 32, folds * 4, iters);
  test_bitonic_topk_fp32(top_64_f32_avx<SORT_INCR>, 64, folds * 2, iters);
  test_bitonic_topk_fp32(top_128_f32_avx<SORT_INCR>, 128, folds, iters);
}


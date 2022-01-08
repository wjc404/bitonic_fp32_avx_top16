#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>
#include <cstdio>

#include "Kernel.hpp"

using std::chrono::steady_clock;

int test_bitonic_topk_fp32(void (*func)(float*, std::size_t),
  const std::size_t k, const std::size_t folds) {

  if (k < 2 || folds < 2) return -1;
  std::cout << "test with k = " << k << " and folds = " << folds << std::endl;
  const std::size_t size = k * folds;
  std::vector<float> ref(size), tst(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1, 1);
  for (std::size_t i = 0; i < size; ++i) {
    ref[i] = tst[i] = dist(gen);
  }
  auto st = steady_clock::now();
  std::nth_element(ref.begin(), ref.begin() + k - 1, ref.end());
  auto et = steady_clock::now();
  std::cout << "nth_element cost = " << std::chrono::duration<double>(
    et - st).count() << " seconds\n";
  std::sort(ref.begin(), ref.begin() + k - 1);
  std::sort(ref.begin() + k, ref.end());

  st = steady_clock::now();
  (*func)(tst.data(), folds);
  et = steady_clock::now();
  std::cout << "bitonic_avx cost = " << std::chrono::duration<double>(
    et - st).count() << " seconds\n";
  std::sort(tst.begin() + k, tst.end());

  for (std::size_t i = 0; i < size; ++i) {
    if (ref[i] != tst[i]) {
      std::cout << "inconsistency at index = " << i << " for k = "
        << k << " and folds = " << folds << std::endl;
      return -1;
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  const std::size_t folds = argc > 1 ? std::atoi(argv[1]) : 300;
  constexpr unsigned int iters = 10;
  for (unsigned int i = 0; i < iters; ++i) {
    if (test_bitonic_topk_fp32(top_16_f32_avx<SORT_INCR>, 16, folds)) break;
  }
  for (unsigned int i = 0; i < iters; ++i) {
    if (test_bitonic_topk_fp32(top_32_f32_avx<SORT_INCR>, 32, folds)) break;
  }
}


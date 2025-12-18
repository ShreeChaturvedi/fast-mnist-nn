#ifndef NEURAL_NET_CPP
#define NEURAL_NET_CPP

/**
 * A simple neural network implementationi n C++.  This implementation
 * is essentially based on the implementation from Michael Nielsen at
 * http://neuralnetworksanddeeplearning.com/
 *
 * Copyright (C) 2021 raodm@miamiOH.edu
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <cstring>

#include "NeuralNet.h"

#if defined(__AVX512F__)
  #define NN_HAS_AVX512 1
#elif defined(__AVX2__)
  #define NN_HAS_AVX2 1
#else
  #define NN_HAS_SCALAR 1
#endif

#if NN_HAS_AVX512
/*
 * Compute dot product of one row with a vector using AVX-512
 * vectorization with dual accumulators. This is the key performance
 * optimization for matrix operations. Uses target attribute to allow
 * AVX-512 intrinsics without requiring global compile flags.
 */
static inline __attribute__((target("avx512f,fma"))) double dot512_rowvec(
    const double* __restrict row,
    const double* __restrict x,
    std::size_t n) {
    std::size_t k = 0, n16 = n & ~std::size_t(15);
    __m512d acc0 = _mm512_setzero_pd(), acc1 = _mm512_setzero_pd();
    for (; k < n16; k += 16) {
        acc0 = _mm512_fmadd_pd(_mm512_load_pd(row + k),
                               _mm512_load_pd(x   + k), acc0);
        acc1 = _mm512_fmadd_pd(_mm512_load_pd(row + k + 8),
                               _mm512_load_pd(x   + k + 8), acc1);
    }
    __m512d acc = _mm512_add_pd(acc0, acc1);
    __m256d lo  = _mm512_castpd512_pd256(acc);
    __m256d hi  = _mm512_extractf64x4_pd(acc, 1);
    __m256d s   = _mm256_add_pd(lo, hi);
    __m128d l   = _mm256_castpd256_pd128(s);
    __m128d h   = _mm256_extractf128_pd(s, 1);
    __m128d p   = _mm_add_pd(l, h);
    double tmp[2]; _mm_storeu_pd(tmp, p);
    double sum = tmp[0] + tmp[1];
    for (; k < n; ++k) sum += row[k] * x[k];  // tail
    return sum;
}

/*
 * SGD weight update for one row using AVX-512. Computes:
 * wr[j] += scale * ap[j] for all j in [0, n).
 */
static inline __attribute__((target("avx512f,fma"))) void sgd_update_row_avx512(
    double* __restrict wr,
    const double* __restrict ap,
    std::size_t n,
    double scale) {
    std::size_t k = 0, n8 = n & ~std::size_t(7);
    __m512d s = _mm512_set1_pd(scale);
    for (; k < n8; k += 8) {
        __m512d w = _mm512_load_pd(wr + k);
        __m512d a = _mm512_load_pd(ap + k);
        _mm512_store_pd(wr + k, _mm512_fmadd_pd(s, a, w));
    }
    for (; k < n; ++k) wr[k] += scale * ap[k];
}
#endif

/*
 * Fused operation: y = sigmoid(W * x + b) where W is a row-major
 * matrix, x is a column vector, and y is the output vector.
 * This combines matrix-vector multiply, bias addition, and sigmoid
 * activation into one optimized operation.
 */
static inline void gemv_rowplusbias_sigmoid(const Matrix& W, const Matrix& b,
                                            const Matrix& x, Matrix& y) {
    const std::size_t m = W.height(), n = W.width();
    const double* __restrict xp = &x[0][0];
    for (std::size_t i = 0; i < m; ++i) {
        const double* __restrict row = W[i].data();
        double s = 0.0;
#if NN_HAS_AVX512
        s = dot512_rowvec(row, xp, n);
#else
        for (std::size_t k = 0; k < n; ++k) s += row[k] * xp[k];
#endif
        s += b[i][0];
        y[i][0] = 1.0 / (1.0 + std::exp(-s));
    }
}

/*
 * Zero out a vector using vectorized operations.
 */
static inline void zero_vector(double* dp, std::size_t n) {
    std::size_t k = 0;
#if NN_HAS_AVX512
    const std::size_t n8 = n & ~std::size_t(7);
    __m512d z = _mm512_setzero_pd();
    for (; k < n8; k += 8) {
        _mm512_store_pd(dp + k, z);
    }
#endif
    for (; k < n; ++k) {
        dp[k] = 0.0;
    }
}

/*
 * Accumulate W^T * delta using row-wise AXPY operations.
 * Computes: dst += delta[i] * W[i,:] for each row i.
 */
static inline void accumulate_wt_delta(const Matrix& W,
                                       const Matrix& delta,
                                       double* __restrict dst) {
    const std::size_t m = W.height(), n = W.width();
    for (std::size_t i = 0; i < m; ++i) {
        const double alpha = delta[i][0];
        if (alpha == 0.0) continue;

        const double* __restrict wr = W[i].data();
        std::size_t k = 0;
#if NN_HAS_AVX512
        const std::size_t n8 = n & ~std::size_t(7);
        __m512d a = _mm512_set1_pd(alpha);
        for (; k < n8; k += 8) {
            __m512d d = _mm512_load_pd(dst + k);
            __m512d w = _mm512_load_pd(wr + k);
            d = _mm512_fmadd_pd(a, w, d);
            _mm512_store_pd(dst + k, d);
        }
#endif
        for (; k < n; ++k) {
            dst[k] += alpha * wr[k];
        }
    }
}

/*
 * Apply sigmoid derivative element-wise: dst *= a * (1 - a).
 */
static inline void apply_sigmoid_derivative(const double* __restrict ap,
                                            double* __restrict dp,
                                            std::size_t n) {
    std::size_t k = 0;
#if NN_HAS_AVX512
    const std::size_t n8 = n & ~std::size_t(7);
    for (; k < n8; k += 8) {
        __m512d a   = _mm512_load_pd(ap + k);
        __m512d one = _mm512_set1_pd(1.0);
        __m512d sp  = _mm512_mul_pd(a, _mm512_sub_pd(one, a));
        __m512d d   = _mm512_load_pd(dp + k);
        _mm512_store_pd(dp + k, _mm512_mul_pd(d, sp));
    }
#endif
    for (; k < n; ++k) {
        const double a = ap[k];
        dp[k] *= a * (1.0 - a);
    }
}

/*
 * Backpropagation: compute delta for previous layer using the formula
 * delta_prev = (W^T * delta) element-multiply (a_prev * (1 - a_prev)).
 * This is optimized for row-major storage by accumulating row-wise
 * AXPY operations.
 */
static inline void backprop_delta_from_rows(const Matrix& W,
                                            const Matrix& delta,
                                            const Matrix& a_prev,
                                            Matrix& delta_prev) {
    const std::size_t n = W.width();
    double* dp = &delta_prev[0][0];
    const double* ap = &a_prev[0][0];

    zero_vector(dp, n);
    accumulate_wt_delta(W, delta, dp);
    apply_sigmoid_derivative(ap, dp, n);
}

/*
 * SGD update: W -= eta * delta * a_prev^T and b -= eta * delta.
 * Performs in-place weight and bias updates using vectorized row-wise
 * AXPY operations.
 */
static inline void sgd_update_inplace(Matrix& W, Matrix& b,
                                      const Matrix& delta,
                                      const Matrix& a_prev,
                                      double eta) {
    const std::size_t m = W.height(), n = W.width();
    const double* __restrict ap = &a_prev[0][0];

    for (std::size_t i = 0; i < m; ++i) {
        const double scale = -eta * delta[i][0];
        // Update bias
        b[i][0] += -eta * delta[i][0];

        // Update weights for this row
        double* __restrict wr = W[i].data();
#if NN_HAS_AVX512
        sgd_update_row_avx512(wr, ap, n, scale);
#else
        for (std::size_t j = 0; j < n; ++j) wr[j] += scale * ap[j];
#endif
    }
}

/*
 * Initialize static buffers for activations and deltas.
 * This is called once and reused for all subsequent training steps.
 */
static inline void init_learn_buffers(std::vector<Matrix>& a,
                                      std::vector<Matrix>& delta,
                                      const MatrixVec& weights,
                                      int L) {
    a.resize(L + 1);
    a[0] = Matrix(weights[0].width(), 1, Matrix::NoInit{});
    for (int l = 1; l <= L; ++l) {
        a[l] = Matrix(weights[l-1].height(), 1, Matrix::NoInit{});
    }
    delta.resize(L + 1);
    for (int l = 1; l <= L; ++l) {
        delta[l] = Matrix(weights[l-1].height(), 1, Matrix::NoInit{});
    }
}

/*
 * Compute output layer delta: (a_L - expected) * sigmoid'(a_L)
 * where sigmoid'(x) = x * (1 - x) for the sigmoid function.
 */
static inline void compute_output_delta(const Matrix& a_L,
                                        const Matrix& expected,
                                        Matrix& delta_L) {
    const std::size_t m = a_L.height();
    double* __restrict d = &delta_L[0][0];
    const double* __restrict aL = &a_L[0][0];
    const double* __restrict y  = &expected[0][0];

    std::size_t k = 0;
#if NN_HAS_AVX512
    const std::size_t m8 = m & ~std::size_t(7);
    for (; k < m8; k += 8) {
        __m512d av = _mm512_load_pd(aL + k);
        __m512d yv = _mm512_load_pd(y  + k);
        __m512d diff = _mm512_sub_pd(av, yv);
        __m512d one  = _mm512_set1_pd(1.0);
        __m512d sp   = _mm512_mul_pd(av, _mm512_sub_pd(one, av));
        _mm512_store_pd(d + k, _mm512_mul_pd(diff, sp));
    }
#endif
    for (; k < m; ++k) {
        const double av = aL[k];
        d[k] = (av - y[k]) * (av * (1.0 - av));
    }
}

/*
 * The constructor to create a neural network with a given number of
 * layers, with each layer having a given number of neurons.
 */
NeuralNet::NeuralNet(const std::vector<int>& layers) :
    layerSizes(1, layers.size()) {
    // Copy the values into the layer size matrix
    std::copy_n(layers.begin(), layers.size(), layerSizes[0].begin());
    // Use helper method to initializes matrices to default values.
    initBiasAndWeightMatrices(layers, biases, weights);
}

/*
 * Helper method called from the constructor to initialize the biases
 * and weight matrices for each layer in the neural netowrk.
 */
void NeuralNet::initBiasAndWeightMatrices(const std::vector<int>& layerSizes,
    MatrixVec& biases, MatrixVec& weights) const {    
    // Create the column matrices for each layer in the nnet.  Each
    // value is initialized with a random value in the range 0 to 1.0
    for (size_t lyr = 1; (lyr < layerSizes.size()); lyr++) {
        // Convenience variables to keep code readable
        const int rows = layerSizes.at(lyr), cols = layerSizes.at(lyr - 1);
        
        biases.push_back(Matrix(rows, 1));

        // Create the 2-D matrices of weights for each layer
        weights.push_back(Matrix(rows, cols));
    }    
}

/*
 * The main learning method that uses fused AVX-512 operations for
 * performing forward pass and backpropagation to update weights and
 * biases for each layer in the neural network.
 */
void NeuralNet::learn(const Matrix& inputs, const Matrix& expected,
                      const Val eta) {
    const int L = static_cast<int>(layerSizes[0].size()) - 1;

    // Use static scratch space to avoid repeated allocations
    static std::vector<Matrix> a;
    static std::vector<Matrix> delta;
    if (a.empty()) init_learn_buffers(a, delta, weights, L);

    // Copy input to activation buffer
    std::memcpy(&a[0][0][0], &inputs[0][0], a[0].height() * sizeof(double));

    // Forward pass: a[l] = sigmoid(W[l-1] * a[l-1] + b[l-1])
    for (int l = 1; l <= L; ++l)
        gemv_rowplusbias_sigmoid(weights[l - 1], biases[l - 1], a[l - 1], a[l]);

    // Compute output layer delta
    compute_output_delta(a[L], expected, delta[L]);

    // Backpropagation and weight updates for each layer
    for (int l = L; l >= 1; --l) {
        if (l > 1) backprop_delta_from_rows(weights[l-1], delta[l],
                                                a[l-1], delta[l-1]);
        sgd_update_inplace(weights[l-1], biases[l-1],
                            delta[l], a[l-1], eta);
    }
}

/*
 * The stream insertion operator to save/write the neural network data
 * to a given file or output stream.
 */
std::ostream& operator<<(std::ostream& os, const NeuralNet& nnet) {
    // First print the layer sizes
    os << nnet.layerSizes << '\n';
    // Next print the biases for each layer.
    for (const auto& bias : nnet.biases) {
        os << bias << '\n';
    }
    // Next print the weights for each layer.
    for (const auto& weight : nnet.weights) {
        os << weight << '\n';
    }
    // Return the output stream as per convention
    return os;
}

/*
 * The stream extraction operator to load neural network data from a
 * given file or input stream.
 */
std::istream& operator>>(std::istream& is, NeuralNet& nnet) {
    // First load the layer sizes
    is >> nnet.layerSizes;
    const int layerCount = nnet.layerSizes[0].size();
    // Now read the biases for each layer
    Matrix temp;
    for (int i = 0; (i < layerCount); i++) {
        is >> temp;
        nnet.biases.push_back(temp);
    }
    // Now read the weights for each layer
    for (int i = 0; (i < layerCount); i++) {
        is >> temp;
        nnet.weights.push_back(temp);
    }
    // Return the input stream as per convention
    return is;
}

/*
 * The method to classify/recognize a given input. Uses static buffers
 * to avoid allocations and fused operations for better performance.
 */
Matrix
NeuralNet::classify(const Matrix& inputs) const {
    // Use static buffers to avoid repeated allocations. This assumes
    // a two-layer network (input, hidden, then output).
    static Matrix hidden(weights[0].height(), 1, Matrix::NoInit{});

    // Forward pass through first layer
    gemv_rowplusbias_sigmoid(weights[0], biases[0], inputs, hidden);

    // Forward pass through second layer and return result
    Matrix out(weights[1].height(), 1, Matrix::NoInit{});
    gemv_rowplusbias_sigmoid(weights[1], biases[1], hidden, out);
    return out;
}

#endif

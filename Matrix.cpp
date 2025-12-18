#ifndef MATRIX_CPP
#define MATRIX_CPP

#include <cassert>
#include <algorithm>
#include <cstring>
#include <iostream>


#include "Matrix.h"


/** Allocate 64B-aligned storage and compute padded leading dimension. */
void Matrix::allocate() {
    if (rows_ == 0 || cols_ == 0) { data_ = nullptr; ld_ = 0; return; }
    static constexpr std::size_t AlignBytes = 64;
    static constexpr std::size_t AlignElem  = AlignBytes / sizeof(Val);
    ld_ = (cols_ == 1) ? 1 : ((cols_ + (AlignElem - 1)) & ~(AlignElem - 1));
    void* p = nullptr;
    if (::posix_memalign(&p, AlignBytes, sizeof(Val) * rows_ * ld_) != 0 || !p)
        throw std::bad_alloc();
    data_ = static_cast<Val*>(p);
}

/** Deallocate internal storage and reset pointer. */
void Matrix::deallocate() {
    std::free(data_);
    data_ = nullptr;
}

/** Copy constructor: deep-copies data honoring per-matrix leading dimension. */
Matrix::Matrix(const Matrix& other)
    : rows_(other.rows_), cols_(other.cols_), ld_(0), data_(nullptr) {
    allocate();  // sets padded ld_ for *this*
    if (rows_ == 0 || cols_ == 0) return;

    if (ld_ == other.ld_) {
        std::memcpy(data_, other.data_, rows_ * ld_ * sizeof(Val));
    } else {
        for (std::size_t r = 0; r < rows_; ++r) {
            std::memcpy(data_ + r*ld_, other.data_ + r*other.ld_,
                        cols_ * sizeof(Val));
        }
    }
}


/** Move constructor: transfers ownership without allocation or copy. */
Matrix::Matrix(Matrix&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), ld_(other.ld_),
        data_(other.data_) {
    other.rows_ = other.cols_ = other.ld_ = 0;
    other.data_ = nullptr;
}

/** Copy assignment: resizes as needed and deep-copies data. */
Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other) return *this;

    // Compute padded ld for target dims
    static constexpr std::size_t AlignElem = 64 / sizeof(Val);
    const std::size_t want_ld = (other.cols_ + (AlignElem - 1))
                                & ~(AlignElem - 1);  // padded ld

    const bool need_realloc = (rows_ != other.rows_) || (cols_ != other.cols_) 
                              || (ld_ != want_ld);

    if (need_realloc) {
        deallocate();
        rows_ = other.rows_;
        cols_ = other.cols_;
        ld_ = 0;
        allocate(); 
    } else {
        rows_ = other.rows_;
        cols_ = other.cols_;
    }  // ld_ unchanged

    if (rows_ == 0 || cols_ == 0) return *this;

    if (ld_ == other.ld_) {  // fast path for same ld_
        std::memcpy(data_, other.data_, rows_ * ld_ * sizeof(Val));
    } else {
        for (std::size_t r = 0; r < rows_; ++r) {  // slow path if different ld_
            std::memcpy(data_ + r*ld_, other.data_ + r * other.ld_,
                        cols_ * sizeof(Val));
        }
    }
    return *this;
}

/** Move assignment: releases current storage and takes ownership. */
Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        deallocate();
        rows_ = other.rows_; cols_ = other.cols_; ld_ = other.ld_;
        data_ = other.data_;
        other.rows_ = other.cols_ = other.ld_ = 0;
        other.data_ = nullptr;
    }
    return *this;
}

/** Destructor: releases internal storage. */
Matrix::~Matrix() {
    deallocate();
}

/** \brief Stream insertion for human-readable matrix output.
 *
 * Prints rows, columns, and elements in row-major order.
 */
std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    // Print the number of rows and columns to ease reading
    os << matrix.height() << " " << matrix.width() << '\n';
    // Print each entry to the output stream.
    for (std::size_t r = 0; r < matrix.rows_; ++r) {
        for (std::size_t c = 0; c < matrix.cols_; ++c) {
            os << matrix[r][c] << " ";
        }
        // Print a new line at the end of each row just to format the
        // output a bit nicely.
        os << '\n';
    }
    return os;
}

/** \brief Stream extraction for reading matrices written by operator<<. */
std::istream& operator>>(std::istream& is, Matrix& matrix) {
    // Temporary variables to load matrix sizes
    std::size_t height, width;
    is >> height >> width;
    // Now initialize the destination matrix to ensure it is of the
    // correct dimension.
    matrix = Matrix(height, width);
    // Read each entry from the input stream.
    for (std::size_t r = 0; r < height; ++r) {
        for (std::size_t c = 0; c < width; ++c) {
            is >> matrix[r][c];
        }
    }
    return is;
}


/** Construct an uninitialized matrix of size row×col. */
Matrix::Matrix(std::size_t row, std::size_t col, NoInit)
    : rows_(row), cols_(col), ld_(0), data_(nullptr) {
    allocate();  // sets ld_ with 64B padding and posix_memalign(64,...)
}

/** Construct and initialize all elements to initVal. */
Matrix::Matrix(std::size_t row, std::size_t col, Val initVal)
    : rows_(row), cols_(col), ld_(0), data_(nullptr) {
    allocate();  // sets padded ld_ and 64B-aligned data_
    if (rows_ == 0 || cols_ == 0) return;

    if (initVal == Val(0)) {
        // fastest for zero: one bulk memset (padding included)
        std::memset(data_, 0, rows_ * ld_ * sizeof(Val));
    } else {
        // only initialize logical cols, leave padded tail alone
        for (std::size_t r = 0; r < rows_; ++r) {
            Val* dst = data_ + r * ld_;
            for (std::size_t c = 0; c < cols_; ++c) dst[c] = initVal;
        }
    }
}

/** Fast path for outer product when inner dimension is one.
 *
 * Multiplies a column vector A by a row vector B and writes the
 * result rows into C. This avoids reductions and reads the row vector
 * once while scaling it by the scalar from each row of A.
 *
 * \param[in] A Pointer to the left matrix data.
 * \param[in] B Pointer to the right matrix data (first row).
 * \param[out] C Pointer to destination matrix data.
 * \param[in] rows Number of output rows.
 * \param[in] cols Number of output columns.
 * \param[in] ldA Leading dimension of A.
 * \param[in] ldC Leading dimension of C.
 */
static inline void dotOuterOne(const Val* __restrict A, const Val* __restrict B,
    Val* __restrict C, std::size_t rows, std::size_t cols, std::size_t ldA,
    std::size_t ldC) {
    for (std::size_t i = 0; i < rows; ++i) {
        // get the scalar from this input row
        const Val aScale = A[i * ldA];
        // write scaled row into the output
        Val* outRow = C + i * ldC;
        for (std::size_t j = 0; j < cols; ++j) outRow[j] = aScale * B[j];
    }
}

/** Multiply and accumulate a single output tile over the inner dimension.
 *
 * This helper updates the output block C[rowBegin:rowEnd, colBegin:colEnd]
 * by iterating the inner dimension in small chunks for cache reuse. The
 * right matrix is supplied as its transpose so each inner loop uses two
 * contiguous slices to improve memory bandwidth.
 *
 * \param[in] A Pointer to left matrix data.
 * \param[in] BT Pointer to right matrix data transposed.
 * \param[out] C Pointer to destination matrix data.
 * \param[in] ldA Leading dimension of A.
 * \param[in] ldBT Leading dimension of BT.
 * \param[in] ldC Leading dimension of C.
 * \param[in] rowBegin First output row in this tile.
 * \param[in] rowEnd One past last output row in this tile.
 * \param[in] colBegin First output column in this tile.
 * \param[in] colEnd One past last output column in this tile.
 * \param[in] innerDim Shared inner dimension K to accumulate over.
 */
// SIMD dot product with 4-wide accumulation and scalar tail
static inline double dotVec4Sum(const Val* __restrict aRow,
                                const Val* __restrict bRow,
                                std::size_t kt) {
    using v4d = double __attribute__((vector_size(32)));
    v4d acc = {0.0, 0.0, 0.0, 0.0};
    std::size_t kk = 0;
    const std::size_t kt4 = kt & ~std::size_t(3);
    for (; kk < kt4; kk += 4) {
        const v4d va = *reinterpret_cast<const v4d*>(aRow + kk);
        const v4d vb = *reinterpret_cast<const v4d*>(bRow + kk);
        acc += va * vb;
    }
    const double* ap = reinterpret_cast<const double*>(&acc);
    double sum = ap[0] + ap[1] + ap[2] + ap[3];
    for (; kk < kt; ++kk) sum += aRow[kk] * bRow[kk];
    return sum;
}

static inline void gemmTileBlock(const Val* __restrict A,
                                 const Val* __restrict BT,
                                 Val* __restrict C,
                                 std::size_t ldA, std::size_t ldBT,
                                 std::size_t ldC,
                                 std::size_t rBegin, std::size_t rEnd,
                                 std::size_t cBegin, std::size_t cEnd,
                                 std::size_t innerDim) {
    // walk K in tiles to keep hot data in cache
    for (std::size_t k0 = 0; k0 < innerDim; k0 += 128) {
        const std::size_t kt = std::min(innerDim, k0 + 128) - k0;
        for (std::size_t i = rBegin; i < rEnd; ++i) {
            const Val* __restrict aRow = A + i * ldA + k0;
            Val*       __restrict cRow = C + i * ldC + cBegin;
            if (k0 == 0) std::fill_n(cRow, cEnd - cBegin, Val(0));
            for (std::size_t j = cBegin; j < cEnd; ++j) {
                const Val* __restrict bRow = BT + j * ldBT + k0;
                const double sum = dotVec4Sum(aRow, bRow, kt);
                cRow[j - cBegin] += sum;
            }
        }
    }
}

/**
 * Performs DGEMM using blocked matrix multiplication.
 * 
 * \param[in] rhs The other matrix to be used.  This matrix must
 * have the same number of rows as the number of columns in this
 * matrix.  Otherwise this method throws an excpetion.
 *
 * \return The resulting matrix in which each value has been
 * computed by multiplying the corresponding values from \c this
 * and rhs.
 * 
 */
Matrix Matrix::dot(const Matrix& rhs) const {
    assert(cols_ == rhs.rows_);
    Matrix result(rows_, rhs.cols_, Matrix::NoInit{});
    if (!rows_ || !cols_ || !rhs.cols_) return result;
    if (cols_ == 1) { dotOuterOne(data_, rhs.data_, result.data_, rows_, 
        rhs.cols_, ld_, result.ld_); return result; }
    Matrix BT = rhs.transpose();
    constexpr std::size_t BI = 64, BJ = 64;
    for (std::size_t i0 = 0; i0 < rows_; i0 += BI) {
        const std::size_t iMax = std::min(rows_, i0 + BI);
        for (std::size_t j0 = 0; j0 < rhs.cols_; j0 += BJ) {
            const std::size_t jMax = std::min(rhs.cols_, j0 + BJ);
            gemmTileBlock(data_, BT.data_, result.data_, ld_, BT.ld_,
                            result.ld_, i0, iMax, j0, jMax, cols_);
        }
    }
    return result;
}

/** Copy a rectangular tile from src to dst while transposing.
 *
 * This helper copies a tile of rows and columns by walking the source
 * column and writing to the destination row. It uses a small unroll to
 * reduce loop overhead for contiguous stores.
 *
 * \param[in] src Pointer to source matrix data.
 * \param[out] dst Pointer to destination matrix data.
 * \param[in] lda Leading dimension of source.
 * \param[in] ldb Leading dimension of destination.
 * \param[in] rowBegin First source row in the tile.
 * \param[in] rowEnd One past last source row in the tile.
 * \param[in] colBegin First source column in the tile.
 * \param[in] colEnd One past last source column in the tile.
 */
static inline void transposeTileCopy(const Val* __restrict src, 
    Val* __restrict dst, std::size_t lda, std::size_t ldb, std::size_t rowBegin,
    std::size_t rowEnd, std::size_t colBegin, std::size_t colEnd) {
    for (std::size_t col = colBegin; col < colEnd; ++col) {
        // set pointers to the input column and output row
        const Val* __restrict s = src + rowBegin * lda + col;
        Val*       __restrict d = dst + col * ldb + rowBegin;
        std::size_t r = rowBegin;
        // unroll by four to cut loop overhead on contiguous stores
        const std::size_t rEnd4 = rowBegin + ((rowEnd - rowBegin) 
                                    & ~std::size_t(3));
        for (; r < rEnd4; r += 4) {
            d[0] = s[0 * lda]; d[1] = s[1 * lda];
            d[2] = s[2 * lda]; d[3] = s[3 * lda];
            s += 4 * lda; d += 4;
        }
        // finish any remaining rows in this tile
        for (; r < rowEnd; ++r) { *d++ = *s; s += lda; }
    }
}

/** Return a transposed copy of this matrix using blocked copies. */
Matrix Matrix::transpose() const {
    if (rows_ == 0 || cols_ == 0) return *this;
    Matrix out(cols_, rows_, Matrix::NoInit{});
    const std::size_t srcRows = rows_, srcCols = cols_;
    const std::size_t lda = ld_, ldb = out.ld_;
    const Val* __restrict src = data_; Val* __restrict dst = out.data_;

    // fast paths for vector shapes
    if (srcRows == 1) {
        for (std::size_t j = 0; j < srcCols; ++j)
            dst[j * ldb] = src[j];
    }
    if (srcCols == 1) {
        for (std::size_t i = 0; i < srcRows; ++i)
            dst[i] = src[i * lda];
    }
    // blocked transpose with contiguous stores on destination
    constexpr std::size_t TileSize = 32;
    for (std::size_t col0 = 0; col0 < srcCols; col0 += TileSize) {
        const std::size_t colEnd = std::min(srcCols, col0 + TileSize);
        for (std::size_t row0 = 0; row0 < srcRows; row0 += TileSize) {
            const std::size_t rowEnd = std::min(srcRows, row0 + TileSize);
            transposeTileCopy(src, dst, lda, ldb, row0, rowEnd, col0, colEnd);
        }
    }
    return out;
}

/** In-place AXPY: this += alpha * X. */
void Matrix::axpy(Val alpha, const Matrix& X) {
    if (rows_ == 0 || cols_ == 0 || alpha == Val(0)) return;
    for (std::size_t r = 0; r < rows_; ++r) {
        Val* dst = data_ + r * ld_;
        const Val* src = X.data_ + r * X.ld_;
        for (std::size_t c = 0; c < cols_; ++c) dst[c] += alpha * src[c];
    }
} 

#endif

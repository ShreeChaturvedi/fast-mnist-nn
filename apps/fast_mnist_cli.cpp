/**
 * A top-level source that exercises the different features of
 * NeuralNet to recognize handwritten digits.  This implementation is
 * essentially based on the implementation from Michael Nielsen at
 * http://neuralnetworksanddeeplearning.com/
 *
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#include "fast_mnist/NeuralNet.h"

/**
 * Cache accessor to avoid a global variable while retaining reuse.
 *
 * \return A reference to the image cache.
 */
static inline std::unordered_map<std::string, Matrix>& imageCache() {
    static std::unordered_map<std::string, Matrix> cache;
    return cache;
}

namespace fs = std::filesystem;

struct PgmBinHeader {
    std::uint32_t magic;
    std::uint32_t rows;
    std::uint32_t cols;
    std::uint32_t reserved;
};

constexpr std::uint32_t kPgmBinMagic = 0x4D4E5047;

/**
 * Skip whitespace and comment lines in a PGM buffer.
 *
 * \param[in,out] p Pointer into the buffer.
 * \param[in] e Pointer to one past the buffer end.
 */
static inline void skipSpaceAndComments(const char*& p, const char* e) {
    while (p < e) {
        while (p < e && static_cast<unsigned char>(*p) <= ' ')
            ++p;
        if (p < e && *p == '#') {
            while (p < e && *p != '\n')
                ++p;
            continue;
        }
        break;
    }
}

/**
 * Helper method to parse an integer from a string.
 *
 * \param[in] p The pointer to the string to be parsed.
 * \param[in] e The pointer to the end of the string.
 * \return The parsed integer.
 */
static inline int parseInt(const char*& p, const char* e) {
    skipSpaceAndComments(p, e);
    int v = 0;
    auto r = std::from_chars(p, e, v);
    if (r.ec != std::errc{})
        throw std::runtime_error("PGM parse error");
    p = r.ptr;
    return v;
}

/**
 * Convert a relative path into a safe cache file name.
 *
 * \param[in] path Relative path to sanitize.
 * \return Sanitized file name that is safe on common filesystems.
 */
static inline std::string sanitizePathForFilename(const std::string& path) {
    std::string out;
    out.reserve(path.size());
    for (char ch : path) {
        if (ch == '/' || ch == '\\' || ch == ':') {
            out.push_back('_');
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

/**
 * Get or create the cache directory under the data root.
 *
 * \param[in] basePath The data root directory.
 * \return Cache directory path as a string.
 */
static inline const std::string& cacheDirForBase(const std::string& basePath) {
    static std::string cachedBase;
    static std::string cachedDir;
    if (cachedDir.empty() || cachedBase != basePath) {
        fs::path dir = fs::path(basePath) / "cache";
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec) {
            throw std::runtime_error("Cache dir error: " + ec.message());
        }
        cachedBase = basePath;
        cachedDir = dir.string();
    }
    return cachedDir;
}

/**
 * Try to load a normalized image from a binary cache file.
 *
 * \param[in] binPath Path to the cache file.
 * \param[out] img Destination matrix.
 * \return True if the cache was loaded successfully.
 */
static inline bool tryLoadBinaryCache(const std::string& binPath, Matrix& img) {
    std::ifstream file(binPath, std::ios::binary);
    if (!file)
        return false;

    PgmBinHeader hdr{};
    file.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!file || hdr.magic != kPgmBinMagic || hdr.rows == 0 || hdr.cols == 0) {
        return false;
    }

    img = Matrix(hdr.rows, hdr.cols, Matrix::NoInit{});
    for (std::uint32_t r = 0; r < hdr.rows; ++r) {
        file.read(reinterpret_cast<char*>(&img[r][0]), hdr.cols * sizeof(Val));
    }
    return static_cast<bool>(file);
}

/**
 * Write a normalized image to a binary cache file.
 *
 * \param[in] binPath Path to the cache file.
 * \param[in] img Source matrix.
 */
static inline void writeBinaryCache(const std::string& binPath,
                                    const Matrix& img) {
    std::ofstream file(binPath, std::ios::binary);
    if (!file)
        return;

    PgmBinHeader hdr{};
    hdr.magic = kPgmBinMagic;
    hdr.rows = static_cast<std::uint32_t>(img.height());
    hdr.cols = static_cast<std::uint32_t>(img.width());
    file.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    for (std::size_t r = 0; r < img.height(); ++r) {
        file.write(reinterpret_cast<const char*>(&img[r][0]),
                   img.width() * sizeof(Val));
    }
}

/** Read entire file into a std::string buffer.
 *
 * Opens the file in binary mode, determines its size, allocates a
 * string of that size, and reads the contents in a single call.
 *
 * \param[in] path Filesystem path to the input file.
 * \return String containing the exact file contents (may be empty).
 */
static inline std::string readFileToString(const std::string& path) {
    // open file once in binary mode
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Unable to read " + path);

    // get file size and allocate exact buffer
    file.seekg(0, std::ios::end);
    const auto sz = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::string buf(sz, '\0');

    // read all bytes into the buffer
    if (sz)
        file.read(buf.data(), buf.size());
    return buf;
}

/** Load a PGM image file into a column vector matrix.
 *
 * Parses an ASCII P2 PGM file and normalizes pixels to [0, 1]. Uses an
 * in-memory cache and a small on-disk binary cache to avoid repeated
 * parsing across runs.
 *
 * \param[in] basePath Base directory for dataset files.
 * \param[in] relPath Path relative to basePath.
 * \return Reference to a cached n-by-1 matrix of pixel values.
 */
const Matrix& loadPGM(const std::string& basePath, const std::string& relPath) {
    const fs::path fullPath = fs::path(basePath) / relPath;
    const std::string fullPathStr = fullPath.string();
    auto& cache = imageCache();
    if (auto it = cache.find(fullPathStr); it != cache.end()) {
        return it->second;
    }

    const std::string& cacheDir = cacheDirForBase(basePath);
    const std::string binName = sanitizePathForFilename(relPath) + ".bin";
    const std::string binPath = (fs::path(cacheDir) / binName).string();

    Matrix img;
    if (!tryLoadBinaryCache(binPath, img)) {
        std::string buf = readFileToString(fullPathStr);
        const char* p = buf.data();
        const char* e = p + buf.size();

        skipSpaceAndComments(p, e);
        if (p + 2 > e || p[0] != 'P' || p[1] != '2') {
            throw std::runtime_error("Unsupported PGM: " + fullPathStr);
        }
        p += 2;
        const int width = parseInt(p, e);
        const int height = parseInt(p, e);
        const int maxVal = parseInt(p, e);
        if (width <= 0 || height <= 0 || maxVal <= 0) {
            throw std::runtime_error("Invalid PGM header: " + fullPathStr);
        }

        const size_t nPix = static_cast<size_t>(width) * height;
        img = Matrix(nPix, 1, Matrix::NoInit{});
        const double inv = 1.0 / static_cast<double>(maxVal);
        for (size_t i = 0; i < nPix; ++i) {
            const int pix = parseInt(p, e);
            img[i][0] = static_cast<double>(pix) * inv;
        }
        writeBinaryCache(binPath, img);
    }

    auto [it, _] = cache.emplace(fullPathStr, std::move(img));
    return it->second;
}

/**
 * Helper method to compute the expected output for a given image.
 * The expected output is determined from the last digit in a given
 * file name.  For example, if the path is test-image-6883_0.pgm, this
 * method extracts the last "0" in the file name and uses that as the
 * expected digit.  It This method returns a 10x1 matrix with the
 * entry corresponding to the given digit to be set to 1.
 *
 * \param[in] path The path to the PGM file from where the digit is extracted.
 */
const Matrix& getExpectedDigitOutput(const std::string& path) {
    static const std::array<Matrix, 10> oneHot = []() {
        std::array<Matrix, 10> labels;
        for (int d = 0; d < 10; ++d) {
            Matrix m(10, 1, 0.0);
            m[d][0] = 1.0;
            labels[d] = std::move(m);
        }
        return labels;
    }();

    const auto labelPos = path.rfind('_');
    if (labelPos == std::string::npos || labelPos + 1 >= path.size()) {
        throw std::runtime_error("Invalid label path: " + path);
    }
    const int label = path[labelPos + 1] - '0';
    if (label < 0 || label > 9) {
        throw std::runtime_error("Invalid label digit: " + path);
    }
    return oneHot[static_cast<std::size_t>(label)];
}

/**
 * Count the number of lines in a text file.
 *
 * \param[in] path Path to the file.
 * \return Number of lines in the file.
 */
static inline std::size_t countLines(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Error reading: " + path);
    }
    std::size_t lines = 0;
    std::string line;
    while (std::getline(file, line))
        ++lines;
    return lines;
}

/**
 * Helper method to use the first \c count number of files to train a
 * given neural network.
 *
 * \param[in,out] net The neural network to be trainined.
 *
 * \param[in] path The prefix path to the location where the training
 * images are actually stored.
 *
 * \param[in] fileNames The list of PGM image file names to be used
 * for training.
 *
 * \param[in] count The number of files in this list ot be used.
 */
void train(NeuralNet& net, const std::string& path,
           const std::vector<std::string>& fileNames, int count = 1e6) {
    for (const auto& imgName : fileNames) {
        const Matrix& img = loadPGM(path, imgName);
        const Matrix& exp = getExpectedDigitOutput(imgName);
        net.learn(img, exp);
        if (count-- <= 0) {
            break;
        }
    }
}

/**
 * The top-level method to train a given neural network used a list of
 * files from a given training set.
 *
 * \param[in,out] net The neural network to be trained.
 *
 * \param[in] path The prefix path to the location where the training
 * images are actually stored.
 *
 * \param[in] limit The number of files to be used to train the network.
 *
 * \param[in] imgListFile The file that contains a list of PGM files
 * to be used.  This method randomly shuffles this list before using
 * \c limit nunber of images for training the supplied \c net.
 */
void train(NeuralNet& net, const std::string& path, const int limit = 1e6,
           const std::string& imgListFile = "TrainingSetList.txt") {
    std::ifstream fileList(imgListFile);
    if (!fileList) {
        throw std::runtime_error("Error reading: " + imgListFile);
    }
    std::vector<std::string> fileNames;
    int count = 0;
    // Load the data from the given image file list.
    for (std::string imgName; std::getline(fileList, imgName) && count < limit;
         count++) {
        fileNames.push_back(imgName);
    }
    // Randomly shuffle the list of file names so that we use a random
    // subset of PGM files for training.
    std::random_device rd;
    std::default_random_engine rg(rd());
    std::shuffle(fileNames.begin(), fileNames.end(), rg);
    // Use the helper method to train
    train(net, path, fileNames, limit);
}

/**
 * Helper method to get the index of the maximum element in a given
 * list. For example maxElemIndex({1, 3, -1, 2}) returns 1.
 *
 * \param[in] vec The vector whose maximum element index is to be
 * returned by this method. This list cannot be empty.
 *
 * \return The index position of the maximum element.
 */
int maxElemIndexCol(const Matrix& col) {
    assert(col.width() == 1);
    int best = 0;
    Val bestVal = col[0][0];
    for (std::size_t r = 1; r < col.height(); ++r) {
        const Val v = col[r][0];
        if (v > bestVal) {
            bestVal = v;
            best = r;
        }
    }
    return best;
}

/**
 * Helper method to determine how well a given neural network has
 * trained used a list of test images.
 *
 * \param[in] net The network to be used for classification.
 *
 * \param[in] path The prefix path to the location where the training
 * images are actually stored.
 *
 * \param[in] imgFileList A text file containing the list of
 * image-file-names to be used for assessing the effectiveness of the
 * supplied \c net.
 */
void assess(NeuralNet& net, const std::string& path,
            const std::string& imgFileList = "TestingSetList.txt") {
    std::ifstream fileList2(imgFileList);
    if (!fileList2) {
        throw std::runtime_error("Error reading " + imgFileList);
    }
    // Check how many of the images are correctly classified by the
    // given given neural network.
    auto passCount = 0, totCount = 0;
    ;
    for (std::string imgName; std::getline(fileList2, imgName); totCount++) {
        const Matrix& img = loadPGM(path, imgName);
        const Matrix& exp = getExpectedDigitOutput(imgName);
        // Have our network classify the image.
        const Matrix res = net.classify(img);
        assert(res.width() == 1);
        assert(res.height() == 10);
        // Find the maximum index positions in exp results to see if
        // they are the same. If they are it is a good
        // result. Otherwise, it is an error.
        const int expIdx = maxElemIndexCol(exp);
        const int resIdx = maxElemIndexCol(res);
        if (expIdx == resIdx) {
            passCount++;
        }
    }
    std::cout << "Correct classification: " << passCount << " ["
              << (passCount * 1.f / totCount) << "% ]\n";
}

/**
 * The main method that trains and assess a neural network using a
 * given subset of training images.
 *
 * \param[in] argc The numebr of command-line arguments.  This program
 * requires one path where training & test images are stored. It
 * optionally accepts up to 4 optional command-line arguments.
 *
 * \param[in] argv The actual command-line argument.
 *     1. The path where training and test images are stored.
 *     2. The first argument is assumed to be the number of images to
 *        be used.
 *     3. Number of ephocs to be used for training. Default is 30.
 *     4. The file containing the list of training images to be
 *        used. By default this parameter is set to
 *        "TrainingSetList.txt".
 *     5. The file containing the list of testing images to be
 *        used. By default this parameter is set to
 *        "TestingSetList.txt".
 */
int main(int argc, char* argv[]) {
    // We definitely need 1 argument for the base-path where image
    // files are stored.
    if (argc < 2) {
        std::cout << "Usage: <ImgPath> [#Train] [#Epocs] [TrainSetList] "
                  << "[TestSetList]\n";
        return 1;
    }
    // Process optional command-line arguments or use default values.
    const int imgCount = (argc > 2 ? std::stoi(argv[2]) : 5000);
    const int epochs = (argc > 3 ? std::stoi(argv[3]) : 10);
    const std::string trainImgs = (argc > 4 ? argv[4] : "TrainingSetList.txt");
    const std::string testImgs = (argc > 5 ? argv[5] : "TestingSetList.txt");

    const std::size_t trainListSize = countLines(trainImgs);
    const std::size_t testListSize = countLines(testImgs);
    imageCache().reserve(trainListSize + testListSize + 1024);

    // Create the neural netowrk
    NeuralNet net({784, 30, 10});
    // Train it in at most 30 epochs.
    for (int i = 0; (i < epochs); i++) {
        std::cout << "-- Epoch #" << i << " --\n";
        std::cout << "Training with " << imgCount << " images...\n";
        const auto startTime = std::chrono::high_resolution_clock::now();
        train(net, argv[1], imgCount, trainImgs);
        assess(net, argv[1], testImgs);
        const auto endTime = std::chrono::high_resolution_clock::now();
        // Compute the timeelapsed for this epoch
        using namespace std::literals;
        std::cout << "Elapsed time = " << ((endTime - startTime) / 1ms)
                  << " milliseconds.\n";
    }
    return 0;
}

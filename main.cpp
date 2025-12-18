/**
 * A top-level source that exercises the different features of
 * NeuralNet to recognize handwritten digits.  This implementation is
 * essentially based on the implementation from Michael Nielsen at
 * http://neuralnetworksanddeeplearning.com/
 *
 * Copyright (C) 2021 raodm@miamiOH.edu
 */

#include <string>
#include <fstream>
#include <vector>
#include <random>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <unordered_map>
#include <charconv>

#include "NeuralNet.h"

/**
 * Cache accessor to avoid a global variable while retaining reuse.
 *
 * \return A reference to the image cache.
 */
static inline std::unordered_map<std::string, Matrix>& imageCache() {
    static std::unordered_map<std::string, Matrix> cache;
    return cache; 
}

/**
 * Helper method to parse an integer from a string.
 *
 * \param[in] p The pointer to the string to be parsed.
 * \param[in] e The pointer to the end of the string.
 * \return The parsed integer.
 */
static inline int parseInt(const char*& p, const char* e) {
    while (p < e && static_cast<unsigned char>(*p) <= ' ') ++p;
    int v = 0;
    auto r = std::from_chars(p, e, v);
    if (r.ec != std::errc{}) throw std::runtime_error("PGM parse error");
    p = r.ptr;
    return v;
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
    if (!file) throw std::runtime_error("Unable to read " + path);
    
    // get file size and allocate exact buffer
    file.seekg(0, std::ios::end);
    const auto sz = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::string buf(sz, '\0');

    // read all bytes into the buffer
    if (sz) file.read(buf.data(), buf.size());
    return buf;
}

/** Load a PGM image file into a column vector matrix.
 *
 * Parses an ASCII P2 PGM file. Pixels are normalized to the range
 * [0, 1] by dividing by the header max value. Uses a small cache to
 * avoid re-reading the same image repeatedly.
 *
 * \param[in] path Filesystem path to the PGM image file.
 * \return n-by-1 matrix of normalized pixel values (row-major order).
 */
Matrix loadPGM(const std::string& path) {
    auto& cache = imageCache();
    // return cached image if already loaded
    if (auto it = cache.find(path); it != cache.end()) return it->second;

    // read file once into memory
    std::string buf = readFileToString(path);
    const char *p = buf.data(), *e = p + buf.size();

    // skip leading whitespace and validate the magic header
    while (p < e && static_cast<unsigned char>(*p) <= ' ') ++p;
    if (p + 2 > e || p[0] != 'P' || p[1] != '2') throw std::runtime_error("Unsupported/invalid PGM: " + path);
    
    // read header fields: width, height, max pixel value
    p += 2; 
    const int width = parseInt(p, e);
    const int height = parseInt(p, e);
    const int maxVal = parseInt(p, e);

    // allocate output column vector and compute normalization factor
    const size_t nPix = static_cast<size_t>(width) * height;
    Matrix img(nPix, 1, Matrix::NoInit{});
    const double inv = 1.0 / static_cast<double>(maxVal);
    
    // parse each pixel and write normalized value
    for (size_t i = 0; i < nPix; ++i) { const int pix = parse_int(p, e); img[i][0] = static_cast<double>(pix) * inv; }
    
    // store in cache and return reference copy
    auto [it, _] = cache.emplace(path, std::move(img));
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
Matrix getExpectedDigitOutput(const std::string& path) {
    // Path is of the form .../data/TrainingSet/test-image-6883_0.pgm
    // We need to get to the last "_n" part and use 'n' as the label.
    const auto labelPos = path.rfind('_') + 1;
    // Now we know the index position of the 1-digit label.  Convert
    // the character to integer for convenience.
    const int label = path[labelPos] - '0';
    // Now create the expected matrix with the just the value
    // corresponding to the label set to 1.0
    Matrix expected(10, 1, 0.0);
    expected[label][0] = 1.0;  // Just label should be 1.0
    return expected;
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
           const std::vector<std::string>& fileNames,
           int count = 1e6) {
    for (const auto& imgName : fileNames) {
        const Matrix img = loadPGM(path + "/" + imgName);
        const Matrix exp = getExpectedDigitOutput(imgName);
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
    for (std::string imgName; std::getline(fileList, imgName) &&
             count < limit; count++) {
        fileNames.push_back(imgName);
    }
    // Randomly shuffle the list of file names so that we use a random
    // subset of PGM files for training.
    std::default_random_engine rg;
    std::shuffle(fileNames.begin(), fileNames.end(),
                 std::default_random_engine());
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
        if (v > bestVal) { bestVal = v; best = r; }
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
    auto passCount = 0, totCount = 0;;
    for (std::string imgName; std::getline(fileList2, imgName); totCount++) {
        const Matrix img = loadPGM(path + "/" + imgName);
        const Matrix exp = getExpectedDigitOutput(imgName);
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
int main(int argc, char *argv[]) {
    // We definitely need 1 argument for the base-path where image
    // files are stored.
    if (argc < 2) {
        std::cout << "Usage: <ImgPath> [#Train] [#Epocs] [TrainSetList] "
                  << "[TestSetList]\n";
        return 1;
    }
    // Process optional command-line arguments or use default values.
    const int imgCount  = (argc > 2 ? std::stoi(argv[2]) : 5000);
    const int epochs    = (argc > 3 ? std::stoi(argv[3]) : 10);    
    const std::string trainImgs = (argc > 4 ? argv[4] : "TrainingSetList.txt");
    const std::string testImgs  = (argc > 5 ? argv[5] : "TestingSetList.txt");

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

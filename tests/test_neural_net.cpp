#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <sstream>

#include "fast_mnist/Matrix.h"
#include "fast_mnist/NeuralNet.h"

namespace {

std::string makeZeroNetStream() {
    Matrix layerSizes(1, 3, 0.0);
    layerSizes[0][0] = 2.0;
    layerSizes[0][1] = 2.0;
    layerSizes[0][2] = 2.0;

    Matrix b1(2, 1, 0.0);
    Matrix b2(2, 1, 0.0);
    Matrix w1(2, 2, 0.0);
    Matrix w2(2, 2, 0.0);

    std::ostringstream os;
    os << layerSizes << '\n';
    os << b1 << '\n';
    os << b2 << '\n';
    os << w1 << '\n';
    os << w2 << '\n';
    return os.str();
}

} // namespace

TEST_CASE("NeuralNet loads from stream", "[neural_net]") {
    NeuralNet net({1, 1});
    std::istringstream is(makeZeroNetStream());
    is >> net;

    Matrix input(2, 1, 0.0);
    input[0][0] = 1.0;
    input[1][0] = 1.0;

    Matrix out = net.classify(input);

    REQUIRE(out.height() == 2);
    REQUIRE(out.width() == 1);
    REQUIRE(out[0][0] == Catch::Approx(0.5));
    REQUIRE(out[1][0] == Catch::Approx(0.5));
}

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

TEST_CASE("NeuralNet stream round trip preserves classify", "[neural_net]") {
    // Build a non-zero net by loading zeros then learning once so
    // serialized weights are not all zero.
    NeuralNet net({2, 2, 2});
    {
        std::istringstream is(makeZeroNetStream());
        is >> net;
    }

    Matrix input(2, 1, 0.0);
    input[0][0] = 1.0;
    input[1][0] = 0.0;

    Matrix expected(2, 1, 0.0);
    expected[0][0] = 1.0;
    expected[1][0] = 0.0;
    net.learn(input, expected, 0.5);

    Matrix before = net.classify(input);

    std::ostringstream os;
    os << net;
    const std::string serialized = os.str();

    NeuralNet loaded({1, 1});
    {
        std::istringstream is(serialized);
        is >> loaded;
    }

    Matrix after = loaded.classify(input);
    REQUIRE(after.height() == before.height());
    REQUIRE(after.width() == before.width());
    REQUIRE(after[0][0] == Catch::Approx(before[0][0]));
    REQUIRE(after[1][0] == Catch::Approx(before[1][0]));

    // Second deserialize must replace, not append, layers. Compare
    // full serialization so extra appended matrices would be visible.
    {
        std::istringstream is(serialized);
        is >> loaded;
    }
    std::ostringstream os2;
    os2 << loaded;
    REQUIRE(os2.str() == serialized);

    Matrix afterTwice = loaded.classify(input);
    REQUIRE(afterTwice.height() == before.height());
    REQUIRE(afterTwice[0][0] == Catch::Approx(before[0][0]));
    REQUIRE(afterTwice[1][0] == Catch::Approx(before[1][0]));
}

TEST_CASE("NeuralNet classify supports one weight layer", "[neural_net]") {
    // Zero weights/biases => sigmoid(0) = 0.5 for every output unit.
    NeuralNet net({3, 2});
    Matrix input(3, 1, 0.0);
    input[0][0] = 1.0;
    input[1][0] = -1.0;
    input[2][0] = 0.5;

    Matrix out = net.classify(input);
    REQUIRE(out.height() == 2);
    REQUIRE(out.width() == 1);
    REQUIRE(out[0][0] == Catch::Approx(0.5));
    REQUIRE(out[1][0] == Catch::Approx(0.5));
}

TEST_CASE("NeuralNet classify supports multiple hidden layers",
          "[neural_net]") {
    // Architecture with two hidden layers: 2 -> 3 -> 4 -> 2.
    NeuralNet net({2, 3, 4, 2});
    Matrix input(2, 1, 0.0);
    input[0][0] = 0.25;
    input[1][0] = 0.75;

    Matrix out = net.classify(input);
    REQUIRE(out.height() == 2);
    REQUIRE(out.width() == 1);
    // Zero init still yields sigmoid(0) = 0.5 after any depth.
    REQUIRE(out[0][0] == Catch::Approx(0.5));
    REQUIRE(out[1][0] == Catch::Approx(0.5));
}

TEST_CASE("NeuralNet learn then classify with three weight layers",
          "[neural_net]") {
    NeuralNet net({2, 3, 2});
    Matrix input(2, 1, 0.0);
    input[0][0] = 1.0;
    input[1][0] = 0.0;
    Matrix expected(2, 1, 0.0);
    expected[0][0] = 1.0;
    expected[1][0] = 0.0;

    Matrix before = net.classify(input);
    REQUIRE(before[0][0] == Catch::Approx(0.5));
    REQUIRE(before[1][0] == Catch::Approx(0.5));

    // Multi-layer forward/backward must run and move outputs away
    // from the zero-init fixed point toward the target label.
    for (int i = 0; i < 20; ++i)
        net.learn(input, expected, 0.5);

    Matrix out = net.classify(input);
    REQUIRE(out.height() == 2);
    REQUIRE(out.width() == 1);
    REQUIRE(out[0][0] > before[0][0]);
    REQUIRE(out[1][0] < before[1][0]);
}

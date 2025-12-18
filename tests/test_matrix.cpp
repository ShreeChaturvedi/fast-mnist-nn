#include <catch2/catch_test_macros.hpp>

#include "fast_mnist/Matrix.h"

TEST_CASE("Matrix initializes and indexes", "[matrix]") {
    Matrix m(2, 3, 0.0);
    m[0][0] = 1.0;
    m[0][1] = 2.0;
    m[0][2] = 3.0;
    m[1][0] = 4.0;
    m[1][1] = 5.0;
    m[1][2] = 6.0;

    REQUIRE(m.height() == 2);
    REQUIRE(m.width() == 3);
    REQUIRE(m[1][2] == 6.0);
}

TEST_CASE("Matrix transpose preserves values", "[matrix]") {
    Matrix m(2, 3, 0.0);
    m[0][0] = 1.0;
    m[0][1] = 2.0;
    m[0][2] = 3.0;
    m[1][0] = 4.0;
    m[1][1] = 5.0;
    m[1][2] = 6.0;

    Matrix t = m.transpose();
    REQUIRE(t.height() == 3);
    REQUIRE(t.width() == 2);
    REQUIRE(t[0][0] == 1.0);
    REQUIRE(t[2][1] == 6.0);
}

TEST_CASE("Matrix dot multiplies small matrices", "[matrix]") {
    Matrix a(2, 3, 0.0);
    Matrix b(3, 2, 0.0);

    a[0][0] = 1.0;
    a[0][1] = 2.0;
    a[0][2] = 3.0;
    a[1][0] = 4.0;
    a[1][1] = 5.0;
    a[1][2] = 6.0;

    b[0][0] = 7.0;
    b[0][1] = 8.0;
    b[1][0] = 9.0;
    b[1][1] = 10.0;
    b[2][0] = 11.0;
    b[2][1] = 12.0;

    Matrix c = a.dot(b);
    REQUIRE(c.height() == 2);
    REQUIRE(c.width() == 2);
    REQUIRE(c[0][0] == Approx(58.0));
    REQUIRE(c[0][1] == Approx(64.0));
    REQUIRE(c[1][0] == Approx(139.0));
    REQUIRE(c[1][1] == Approx(154.0));
}

TEST_CASE("Matrix axpy updates in place", "[matrix]") {
    Matrix x(2, 2, 1.0);
    Matrix y(2, 2, 2.0);

    x.axpy(2.0, y);
    REQUIRE(x[0][0] == Approx(5.0));
    REQUIRE(x[1][1] == Approx(5.0));
}

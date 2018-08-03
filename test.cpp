#include <blitz/array.h>
#include <iostream>

using namespace blitz;

int main()
{
    // Create two 4x4 arrays.  We want them to look like matrices, so
    // we'll make the valid index range 1..4 (rather than 0..3 which is
    // the default).

    Range r(1,4);
    Array<float,2> A(r,r), B(r,r);

    // The first will be a Hilbert matrix:
    //
    // a   =   1
    //  ij   -----
    //       i+j-1
    //
    // Blitz++ provides a set of types { firstIndex, secondIndex, ... }
    // which act as placeholders for indices.  These can be used directly
    // in expressions.  For example, we can fill out the A matrix like this:

    firstIndex i;    // Placeholder for the first index
    secondIndex j;   // Placeholder for the second index

    A = 1.0 / (i+j-1);

    cout << "A = " << A << endl;

    // A = 4 x 4
    //         1       0.5  0.333333      0.25
    //       0.5  0.333333      0.25       0.2
    //  0.333333      0.25       0.2  0.166667
    //      0.25       0.2  0.166667  0.142857

    // Now the A matrix has each element equal to a_ij = 1/(i+j-1).

    // The matrix B will be the permutation matrix
    //
    // [ 0 0 0 1 ]
    // [ 0 0 1 0 ]
    // [ 0 1 0 0 ]
    // [ 1 0 0 0 ]
    //
    // Here are two ways of filling out B:

    B = (i == (5-j));         // Using an equation -- a bit cryptic

    cout << "B = " << B << endl;

    // B = 4 x 4
    //         0         0         0         1
    //         0         0         1         0
    //         0         1         0         0
    //         1         0         0         0

    B = 0, 0, 0, 1,           // Using an initializer list
        0, 0, 1, 0,
        0, 1, 0, 0,
        1, 0, 0, 0;

    cout << "B = " << B << endl;

    // Now some examples of tensor-like notation.

    Array<float,3> C(r,r,r);  // A three-dimensional array: 1..4, 1..4, 1..4

    thirdIndex k;             // Placeholder for the third index

    // This expression will set
    //
    // c    = a   * b
    //  ijk    ik    kj

 //   C = A(i,k) * B(k,j);
   cout << "C = " << C << endl;


    // In real tensor notation, the repeated k index would imply a
    // contraction (or summation) along k.  In Blitz++, you must explicitly
    // indicate contractions using the sum(expr, index) function:

    Array<float,2> D(r,r);
    D = sum(A(i,k) * B(k,j), k);//指标收缩, 计算矩阵积

    // The above expression computes the matrix product of A and B.

    cout << "D = " << D << endl;

    // D = 4 x 4
    //      0.25  0.333333       0.5         1
    //       0.2      0.25  0.333333       0.5
    //  0.166667       0.2      0.25  0.333333
    //  0.142857  0.166667       0.2      0.25

    // Indices like i,j,k can be used in any order in an expression.
    // For example, the following computes a kronecker product of A and B,
    // but permutes the indices along the way:

    Array<float,4> E(r,r,r,r);    // A four-dimensional array
    fourthIndex l;                // Placeholder for the fourth index

    E = A(l,j) * B(k,i);//指标轮换
//cout << "E = " << E << endl;


    // Now let's fill out a two-dimensional array with a radially symmetric
    // decaying sinusoid.

    int N = 64;                   // Size of array: N x N
    Array<float,2> F(N,N);
    float midpoint = (N-1)/2.;
    int cycles = 3;
    float omega = 2.0 * M_PI * cycles / double(N);
    float tau = - 10.0 / N;

    F = cos(omega * sqrt(pow2(i-midpoint) + pow2(j-midpoint)))
        * exp(tau * sqrt(pow2(i-midpoint) + pow2(j-midpoint)));
//cout << "F = " << F << endl;

    return 0;
}

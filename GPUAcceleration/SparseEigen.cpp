#include "Functions.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <SymEigsSolver.h>
#include <MatOp/SparseSymMatProd.h>

using namespace Spectra;

__declspec(dllexport) void SparseEigen(int* sparsei, int* sparsej, double* sparsevalues, int nsparse, int sidelength, int nvectors, double* eigenvectors, double* eigenvalues)
{
    Eigen::setNbThreads(16);

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nsparse);
    for (int n = 0; n < nsparse; n++)
        triplets.push_back(Eigen::Triplet<double>(sparsei[n], sparsej[n], sparsevalues[n]));

    Eigen::SparseMatrix<double> M(sidelength, sidelength);
    M.setFromTriplets(triplets.begin(), triplets.end());

    // Construct matrix operation object using the wrapper class SparseGenMatProd
    SparseSymMatProd<double> op(M);

    // Construct eigen solver object, requesting the largest three eigenvalues
    SymEigsSolver< double, SMALLEST_MAGN, SparseSymMatProd<double> > eigs(&op, nvectors, nvectors * 4);

    // Initialize and compute
    eigs.init();
    int nconv = eigs.compute(300, 1e-10, SMALLEST_ALGE);

    // Retrieve results
    if (eigs.info() == SUCCESSFUL)
    {        
        Eigen::VectorXd evalues = eigs.eigenvalues();
        Eigen::MatrixXd evectors = eigs.eigenvectors();

        for (int i = 0; i < nvectors; i++)
        {
            eigenvalues[i] = evalues(i);

            for (int j = 0; j < sidelength; j++)
                eigenvectors[i * sidelength + j] = evectors(j, i);
        }
    }
}
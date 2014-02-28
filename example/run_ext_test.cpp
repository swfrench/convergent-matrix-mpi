#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cassert>

#include <mpi.h>

#include "cm_mpi.hpp"

/**
 * Define test config
 */

#define NITER_MIN 10
#define NITER_MAX 10

// 100000 for 8 x 8
//#define MSUB_REPS 10
//#define MSUB_SIZE 10000
//#define M ( MSUB_REPS * MSUB_SIZE )
//#define MSUB_MIN 1000
//#define MSUB_MAX 1200

// 25000 for 2 x 2
//#define MSUB_REPS 10
//#define MSUB_SIZE 2500
//#define M ( MSUB_REPS * MSUB_SIZE )
//#define MSUB_MIN 250
//#define MSUB_MAX 300

// 10000 for 2 x 2
#define MSUB_REPS 10
#define MSUB_SIZE 1000
#define M ( MSUB_REPS * MSUB_SIZE )
#define MSUB_MIN 100
#define MSUB_MAX 120

#include "tests/test01.hpp"

/**
 * Done w/ test config
 */

// block-cyclic distribution
#define NPCOL 2
#define NPROW 2
#define MB 64
#define NB 64
#define LLD 12544
typedef cm::ConvergentMatrix<double,NPROW,NPCOL,MB,NB,LLD> cmat_t;

using namespace std;

int
main( int argc, char **argv )
{
  cmat_t *dist_mat;
  int niter;
  int rank;
  long *nxs, **ixs;
  double *data;
  double *local_data;

  MPI_Init( &argc, &argv );

  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  // generate test case
  niter = gen_test01<double, long>( rank, &nxs, &ixs, &data );

  printf( "%4i : generated %i rounds of fake update indexing\n",
          rank, niter ); fflush( stdout );

  // init distributed matrix object (block-cyclic: see convergent_matrix.hpp)
  dist_mat = new cmat_t( M, M );
#ifdef ENABLE_CONSISTENCY_CHECK
  dist_mat->consistency_check_on();
#endif

  printf( "%4i : starting update rounds ...\n", rank ); fflush( stdout );

  double wt_tot = 0.0;

  // perform a number of dummy updates
  for ( int r = 0; r < niter; r++ )
    {
      cm::LocalMatrix<double> *GtG;
      GtG = new cm::LocalMatrix<double>( nxs[r], nxs[r], data );
      // track update time
      wt_tot -= MPI_Wtime();
      dist_mat->update( GtG, ixs[r] );
      wt_tot += MPI_Wtime();
      delete GtG;
    }

  // commit all updates to the ConvergentMatrix abstraction
  printf( "%4i : committing ...\n", rank ); fflush( stdout );
  // track commit time
  wt_tot -= MPI_Wtime();
  dist_mat->commit();
  wt_tot += MPI_Wtime();
  printf( "%4i : total time spent in update / commit %fs\n", rank, wt_tot ); fflush( stdout );

  // test the write functionality
#ifdef ENABLE_MPIIO_SUPPORT
  //dist_mat->save( "test.matrix" );
#endif

  // fetch the local PBLAS-compatible block-cyclic storage array
  local_data = dist_mat->get_local_data();
  printf( "%4i : local_data[0] = %f\n", rank, local_data[0] );
  //printf( "%4i : dist_mat( 64, 64 ) = %f\n", rank, (*dist_mat)( 64, 64 ) );
  MPI_Barrier( MPI_COMM_WORLD );

  // safe to delete dist_mat now
  delete dist_mat;

  MPI_Finalize();

  return 0;
}

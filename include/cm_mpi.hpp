/**
 * \mainpage ConvergentMatrixMPI
 *
 * MPI-based alternative implementation of `ConvergentMatrix` - a distributed
 * dense matrix data structure (the latter based in UPC++, a set of PGAS
 * extensions to the C++ language).
 *
 * In lieu of the combination of remote memory management, one-sided bulk copy,
 * and asynchronous remote task execution employed by `ConvergentMatrix`, here
 * we achieve the same functionality using MPI-3 RMA - namely, `MPI_Accumulate`
 * with passive-target locking.
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cassert>

#include <algorithm>

#ifdef ENABLE_CONSISTENCY_CHECK
#include <cmath>
#endif

#include <mpi.h>

// LocalMatrix<T>
#include "local_matrix.hpp"

// Bin<T>
#include "bin_mpi.hpp"

// default bin-size threshold (number of elems) before it is flushed
#ifndef DEFAULT_BIN_FLUSH_THRESHOLD
#define DEFAULT_BIN_FLUSH_THRESHOLD 10000
#endif

/**
 * Contains classes associated with the convergent-matrix abstraction
 */
namespace cm {

  /// @cond INTERNAL_DOCS

  /**
   * Wrapper for \c std::rand() for use in \c std::random_shuffle() (called in
   * \c permute() below).
   */
  int
  rgen( int n )
  {
    return std::rand() % n;
  }

  /**
   * Apply a random permutation (in place) to the supplied array.
   * \param xs Array of type \c int
   * \param nx Length of array \c xs
   *
   * Seeds std::rand() in a manner that should be ok even if all threads hit
   * \c permute() at nearly the same time.
   */
  void
  permute( int *xs, int n )
  {
    int rank;
    unsigned seed;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    seed = ( 1 + rank ) * std::time( NULL );
    std::srand( seed );
    std::random_shuffle( xs, xs + n, rgen );
  }

  /// @endcond

  /**
   * MPI-based convergent matrix abstraction
   * \tparam T Matrix data type (e.g. float)
   * \tparam NPROW Number of rows in the distributed process grid
   * \tparam NPCOL Number of columns in the distributed process grid
   * \tparam MB Distribution blocking factor (leading dimension)
   * \tparam NB Distribution blocking factor (trailing dimension)
   * \tparam LLD Leading dimension of local storage (same on all threads)
   */
  template <typename T,              // matrix type
            long NPROW, long NPCOL,  // pblas process grid
            long MB, long NB,        // pblas local blocking factors
            long LLD>                // pblas local leading dim
  class ConvergentMatrix
  {

   private:

    long _m, _n;
    long _myrow, _mycol;
    long _m_local, _n_local;
    int _mpi_rank, _mpi_size;
    int _bin_flush_threshold;
    int *_bin_flush_order;
    T *_local_ptr;
    Bin<T> **_update_bins;
#ifdef ENABLE_CONSISTENCY_CHECK
    bool _consistency_mode;
    LocalMatrix<T> * _update_record;
#endif
    MPI_Win _target_window;
    MPI_Datatype _mpi_data_type;

    inline void
    flush( int thresh = 0 )
    {
      // flush the bins
      for ( int b = 0; b < _mpi_size; b++ ) {
        int tid = _bin_flush_order[b];
        if ( _update_bins[tid]->size() > thresh )
          _update_bins[tid]->flush();
      }
    }

    inline long
    roc( long m,   // matrix dimension
         long np,  // dimension of the process grid
         long ip,  // index in the process grid
         long mb ) // blocking factor
    {
      long nblocks, m_local;
      // number of full blocks to be distributed
      nblocks = m / mb;
      // set lower bound on required local dimension
      m_local = ( nblocks / np ) * mb;
      // edge cases ...
      if ( nblocks % np > ip )
        m_local += mb; // process ip receives another full block
      else if ( nblocks % np == ip )
        m_local += m % mb; // process ip _may_ receive a partial block
      return m_local;
    }

#ifdef ENABLE_CONSISTENCY_CHECK

    inline void
    sum_updates( T *updates, T *summed_updates )
    {
      MPI_Allreduce( updates, summed_updates, _m * _n, _mpi_data_type, MPI_SUM,
                     MPI_COMM_WORLD );
    }

    inline void
    consistency_check( T *updates )
    {
      long ncheck = 0;
      const T rtol = 1e-8;
      T * summed_updates;

      // sum the recorded updates across threads
      summed_updates = new T [_m * _n];
      sum_updates( updates, summed_updates );

      // ensure the locally-owned data is consistent with the record
      printf( "[%s] Rank %4i : Consistency check start ...\n", __func__,
              _mpi_rank );
      for ( long j = 0; j < _n; j++ )
        if ( ( j / NB ) % NPCOL == _mycol ) {
          long off_j = LLD * ( ( j / ( NB * NPCOL ) ) * NB + j % NB );
          for ( long i = 0; i < _m; i++ )
            if ( ( i / MB ) % NPROW == _myrow ) {
              long ij = off_j + ( i / ( MB * NPROW ) ) * MB + i % MB;
              T rres;
              if ( summed_updates[i + _m * j] == 0.0 )
                rres = 0.0;
              else
                rres = std::abs( ( summed_updates[i + _m * j] - _local_ptr[ij] )
                                 / summed_updates[i + _m * j] );
              assert( rres < rtol );
              ncheck += 1;
            }
        }
      delete [] summed_updates;
      printf( "[%s] Rank %4i : Consistency check PASSED for %li local"
              " entries\n", __func__, _mpi_rank, ncheck );
    }

#endif // ENABLE_CONSISTENCY_CHECK

   public:

    /**
     * The ConvergentMatrix distributed matrix abstraction.
     * \param m Global leading dimension
     * \param n Global trailing dimension
     */
    ConvergentMatrix( long m, long n ) :
      _m(m), _n(n)
    {
      int mpi_init;

      // checks on matrix dimension
      assert( _m > 0 );
      assert( _n > 0 );

      // make sure MPI is already initialized
      assert( MPI_Initialized( &mpi_init ) == MPI_SUCCESS );
      assert( mpi_init );

      // get process pool config
      MPI_Comm_rank( MPI_COMM_WORLD, &_mpi_rank );
      MPI_Comm_size( MPI_COMM_WORLD, &_mpi_size );

      // check on block-cyclic distribution
      assert( NPCOL * NPROW == _mpi_size );

      // setup block-cyclic distribution
      _myrow = _mpi_rank / NPROW;
      _mycol = _mpi_rank % NPCOL;
      // calculate minimum req'd local dimensions
      _m_local = roc( _m, NPROW, _myrow, MB );
      _n_local = roc( _n, NPCOL, _mycol, NB );

      // ensure local storage is of nonzero size
      assert( _m_local > 0 );
      assert( _n_local > 0 );

      // allocate local storage, set up windows and bins

      // check minimum local leading dimension
      assert( _m_local <= LLD );

      // allocate and zero storage
#ifdef NO_MPI_ALLOC_MEM
      _local_ptr = new T[LLD * _n_local];
#else
      MPI_Alloc_mem( LLD * _n_local * sizeof(T), MPI_INFO_NULL, &_local_ptr );
#endif
      std::fill( _local_ptr, _local_ptr + LLD * _n_local, 0 );

      // set flush threashold for bins
      _bin_flush_threshold = DEFAULT_BIN_FLUSH_THRESHOLD;

      // infer mpi datatype of T (using infer_mpi_type() from bin_mpi.hpp)
      _mpi_data_type = infer_mpi_type( _local_ptr );

      // initialize window
      MPI_Win_create( _local_ptr, LLD * _n_local * sizeof(T), sizeof(T),
                      MPI_INFO_NULL, MPI_COMM_WORLD, &_target_window );

      // set up bins
      _update_bins = new Bin<T> * [_mpi_size];
      for ( int tid = 0; tid < _mpi_size; tid++ )
        _update_bins[tid] = new Bin<T>( tid, _target_window );
      MPI_Barrier( MPI_COMM_WORLD );

      // set up random flushing order
      _bin_flush_order = new int [_mpi_size];
      for ( int tid = 0; tid < _mpi_size; tid++ )
        _bin_flush_order[tid] = tid;
      permute( _bin_flush_order, _mpi_size );

      // consistency check is off by default
#ifdef ENABLE_CONSISTENCY_CHECK
      _consistency_mode = false;
      _update_record = NULL;
#endif
    }

    /**
     * Destructor for \c ConvergentMatrix
     *
     * On destruction, will:
     * - Free storage associated with update bins
     * - Free storage associated with the distributed matrix
     */
    ~ConvergentMatrix()
    {
      // clean up bins
      for ( int tid = 0; tid < _mpi_size; tid++ )
        delete _update_bins[tid];
      delete [] _update_bins;
      delete [] _bin_flush_order;

      // free local storage
#ifdef NO_MPI_ALLOC_MEM
      delete [] _local_ptr;
#else
      MPI_Free_mem( _local_ptr );
#endif
    }

    /**
     * Get a raw pointer to the local distributed matrix storage (can be passed
     * to, for example, PBLAS routines).
     *
     * \b Note: The underlying storage _will_ be freed in the
     * \c ConvergentMatrix destructor - for a persistent copy, see
     * \c get_local_data_copy().
     */
    inline T *
    get_local_data() const
    {
      return _local_ptr;
    }

    /**
     * Get a point to a _copy_ of the local distributed matrix storage (can be
     * passed to, for example, PBLAS routines).
     *
     * \b Note: The underlying storage will _not_ be freed in the
     * \c ConvergentMatrix destructor, in contrast to that from
     * \c get_local_data().
     */
    inline T *
    get_local_data_copy() const
    {
      T * copy_ptr = new T[LLD * _n_local];
      std::copy( _local_ptr, _local_ptr + LLD * _n_local, copy_ptr );
      return copy_ptr;
    }

    /**
     * Reset the distributed matrix (implicit barrier). Zeros the associated
     * local storage (as well as the update record if consistency checks are
     * turned on).
     */
    inline void
    reset()
    {
      // synchronize and ensure all updates from previous phase have completed
      commit();

      // zero local storage
      std::fill( _local_ptr, _local_ptr + LLD * _n_local, 0 );

#ifdef ENABLE_CONSISTENCY_CHECK
      // reset consistency check ground truth as well
      if ( _consistency_mode )
        (*_update_record) = (T) 0;
#endif
      // must be called by all threads
      MPI_Barrier( MPI_COMM_WORLD );
    }

    /**
     * Get the flush threshold (maximum bulk-update bin size before a bin is
     * flushed and applied to its target).
     */
    inline int
    bin_flush_threshold() const
    {
      return _bin_flush_threshold;
    }

    /**
     * Set the flush threshold (maximum bulk-update bin size before a bin is
     * flushed and applied to its target).
     * \param thresh The bin-size threshold
     */
    inline void
    bin_flush_threshold( int thresh )
    {
      _bin_flush_threshold = thresh;
    }

    /**
     * Distributed matrix leading dimension
     */
    inline long
    m() const
    {
      return _m;
    }

    /**
     * Distributed matrix trailing dimension
     */
    inline long
    n() const
    {
      return _n;
    }

    /**
     * Process grid row index of this thread
     */
    inline long
    pgrid_row() const
    {
      return _myrow;
    }

    /**
     * Process grid column index of this thread
     */
    inline long
    pgrid_col() const
    {
      return _mycol;
    }

    /**
     * Minimum required leading dimension of local storage - must be less than
     * or equal to template parameter LLD
     */
    inline long
    m_local() const
    {
      return _m_local;
    }

    /**
     * Minimum required trailing dimension of local storage
     */
    inline long
    n_local() const
    {
      return _n_local;
    }

    /**
     * Remote random access (read only) to distributed matrix elements
     * \param ix Leading dimension index
     * \param jx Trailing dimension index
     */
    inline T
    operator()( long ix, long jx )
    {
      // infer thread id and linear index
      int tid = ( jx / NB ) % NPCOL + NPCOL * ( ( ix / MB ) % NPROW );
      long ij = LLD * ( ( jx / ( NB * NPCOL ) ) * NB + jx % NB ) +
                        ( ix / ( MB * NPROW ) ) * MB + ix % MB;
      // should be safe to use either locking construct (only other Gets and
      // Accumulates potentially happening at the same time, with the latter
      // being atompic w.r.t. the target element)
#ifdef USE_MPI_LOCK_SHARED
      MPI_Win_lock( MPI_LOCK_SHARED, tid, 0, _target_window );
#else
      MPI_Win_lock( MPI_LOCK_EXCLUSIVE, tid, 0, _target_window );
#endif
      T elem;
      MPI_Get( &elem,   1, _mpi_data_type,
               tid, ij, 1, _mpi_data_type,
               _target_window );
      MPI_Win_unlock( tid, _target_window );
      return elem;
    }

    /**
     * Distributed matrix update: general case
     * \param Mat The update (strided) slice
     * \param ix Maps slice into distributed matrix (leading dimension)
     * \param jx Maps slice into distributed matrix (trailing dimension)
     */
    void
    update( LocalMatrix<T> *Mat, long *ix, long *jx )
    {
      // bin the local update
      for ( long j = 0; j < Mat->n(); j++ ) {
        int pcol = ( jx[j] / NB ) % NPCOL;
        long off_j = LLD * ( ( jx[j] / ( NB * NPCOL ) ) * NB + jx[j] % NB );
        for ( long i = 0; i < Mat->m(); i++ ) {
          int tid = pcol + NPCOL * ( ( ix[i] / MB ) % NPROW );
          long ij = off_j + ( ix[i] / ( MB * NPROW ) ) * MB + ix[i] % MB;
          _update_bins[tid]->append( (*Mat)( i, j ), ij );
        }
      }
#ifdef ENABLE_CONSISTENCY_CHECK
      if ( _consistency_mode )
        for ( long j = 0; j < Mat->n(); j++ )
          for ( long i = 0; i < Mat->m(); i++ )
            (*_update_record)( ix[i], jx[j] ) += (*Mat)( i, j );
#endif

      // possibly flush bins
      flush( _bin_flush_threshold );
    }

    /**
     * Distributed matrix update: symmetric case
     * \param Mat The update (strided) slice
     * \param ix Maps slice into distributed matrix (both dimensions)
     */
    void
    update( LocalMatrix<T> *Mat, long *ix )
    {
#ifndef NOCHECK
      // must be square to be symmetric
      assert( Mat->m() == Mat->n() );
#endif
      // bin the local update
      for ( long j = 0; j < Mat->n(); j++ ) {
        int pcol = ( ix[j] / NB ) % NPCOL;
        long off_j = LLD * ( ( ix[j] / ( NB * NPCOL ) ) * NB + ix[j] % NB );
        for ( long i = 0; i < Mat->m(); i++ )
          if ( ix[i] <= ix[j] ) {
            int tid = pcol + NPCOL * ( ( ix[i] / MB ) % NPROW );
            long ij = off_j + ( ix[i] / ( MB * NPROW ) ) * MB + ix[i] % MB;
            _update_bins[tid]->append( (*Mat)( i, j ), ij );
          }
      }
#ifdef ENABLE_CONSISTENCY_CHECK
      if ( _consistency_mode )
        for ( long j = 0; j < Mat->n(); j++ )
          for ( long i = 0; i < Mat->m(); i++ )
            if ( ix[i] <= ix[j] )
              (*_update_record)( ix[i], ix[j] ) += (*Mat)( i, j );
#endif

      // possibly flush bins
      flush( _bin_flush_threshold );
    }

    /**
     * Drain all update bins and synchronize. If the consistency check is
     * turned on, it will run after all updates have been applied.
     */
    inline void
    commit()
    {
      // flush all non-empty bins (local task queue will be emptied)
      flush();

      // synchronize
      MPI_Barrier( MPI_COMM_WORLD );

      // if enabled, the consistency check should only occur after commit
#ifdef ENABLE_CONSISTENCY_CHECK
      if ( _consistency_mode )
        consistency_check( _update_record->data() );
#endif
    }

#ifdef ENABLE_CONSISTENCY_CHECK

    /**
     * Turn on consistency check mode (requires compilation with
     * ENABLE_CONSISTENCY_CHECK).
     * NOTE: MPI must be initialized in order for the consistency check to run
     * on calls to commit() (necessary for summation of the replicated update
     * records).
     */
    inline void
    consistency_check_on()
    {
      _consistency_mode = true;
      if ( _update_record == NULL )
        _update_record = new LocalMatrix<T>( _m, _n );
      (*_update_record) = (T) 0;
      printf( "[%s] Rank %4i : Consistency check mode ON (recording ...)\n",
              __func__, _mpi_rank );
    }

    /**
     * Turn off consistency check mode (requires compilation with
     * ENABLE_CONSISTENCY_CHECK).
     */
    inline void
    consistency_check_off()
    {
      _consistency_mode = false;
      if ( _update_record != NULL )
        delete _update_record;
      printf( "[%s] Rank %4i : Consistency check mode OFF\n",
              __func__, _mpi_rank );
    }

#endif // ENABLE_CONSISTENCY_CHECK

#ifdef ENABLE_MPIIO_SUPPORT

    /**
     * Save the distributed matrix to disk via MPI-IO (requres compilation
     * with ENABLE_MPIIO_SUPPORT). No implicit commit() before matrix data is
     * written - always call commit() first.
     * \param fname File name for matrix
     */
    void
    save( const char *fname )
    {
      int mpi_init, mpi_rank, distmat_size, write_count;
      double wt_io, wt_io_max;
      MPI_Status status;
      MPI_Datatype distmat;
      MPI_File f_ata;

      // make sure we all get here
      MPI_Barrier( MPI_COMM_WORLD );

      // check process grid ordering
      MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
      assert( _myrow * NPCOL + _mycol == mpi_rank );

      // initialize distributed type
      int gsizes[]   = { (int)_m, (int)_n },
          distribs[] = { MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC },
          dargs[]    = { MB, NB },
          psizes[]   = { NPROW, NPCOL },
          base_dtype = _mpi_data_type;
      MPI_Type_create_darray( NPCOL * NPROW, mpi_rank, 2,
                              gsizes, distribs, dargs, psizes,
                              MPI_ORDER_FORTRAN,
                              base_dtype,
                              &distmat );
      MPI_Type_commit( &distmat );

      // sanity check on check on distributed array data size
      MPI_Type_size( distmat, &distmat_size );
      assert( distmat_size / sizeof(T) == ( _m_local * _n_local ) );

      // open read-only
      MPI_File_open( MPI_COMM_WORLD, fname,
                    MPI_MODE_CREATE | MPI_MODE_WRONLY,
                    MPI_INFO_NULL, &f_ata );

      // set view w/ distmat
      MPI_File_set_view( f_ata, 0, base_dtype, distmat, "native",
                         MPI_INFO_NULL );

      // compaction in place
      if ( _m_local < LLD )
        for ( long j = 1; j < _n_local; j++ )
          for ( long i = 0; i < _m_local; i++ )
            _local_ptr[i + j * _m_local] = _local_ptr[i + j * LLD];

      // write out local data
      wt_io = - MPI_Wtime();
      MPI_File_write_all( f_ata, _local_ptr, _m_local * _n_local, base_dtype,
                          &status );
      wt_io = wt_io + MPI_Wtime();

      // close; report io time
      MPI_File_close( &f_ata );
      MPI_Reduce( &wt_io, &wt_io_max, 1, MPI_DOUBLE, MPI_MAX, 0,
                  MPI_COMM_WORLD );
      if ( mpi_rank == 0 )
        printf( "[%s] max time spent in matrix write: %.3f s\n", __func__,
                wt_io_max );

      // sanity check on data written
      MPI_Get_count( &status, base_dtype, &write_count );
      assert( write_count == ( _m_local * _n_local ) );

      // expansion in place
      if ( _m_local < LLD )
        for ( long j = _n_local - 1; j > 0; j-- )
          for ( long i = _m_local - 1; i >= 0; i-- )
            _local_ptr[i + j * LLD] = _local_ptr[i + j * _m_local];

      // free distributed type
      MPI_Type_free( &distmat );
    }

#endif // ENABLE_MPIIO_SUPPORT

  }; // end of ConvergentMatrix


}

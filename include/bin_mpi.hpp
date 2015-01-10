#pragma once

#ifndef NOCHECK
#include <cassert>
#endif

#include <vector>
#include <mpi.h>

namespace cm {

  inline MPI_Datatype
  infer_mpi_type( float * )
  {
    return MPI_FLOAT;
  }

  inline MPI_Datatype
  infer_mpi_type( double * )
  {
    return MPI_DOUBLE;
  }

  template <typename T>
  class Bin {

   private:

    int _target_rank;

    MPI_Win _target_window;
    MPI_Datatype _mpi_data_type;

    std::vector<T> _update_data;
    std::vector<long> _update_ijs;

    inline void
    clear()
    {
      _update_data.clear();
      _update_ijs.clear();
    }

   public:

    /**
     * Initialize the Bin object for a given target
     * \param target_rank Rank of the target
     * \param target_window Window for target-local storage
     */
    Bin( int target_rank, MPI_Win target_window ) :
      _target_rank(target_rank), _target_window(target_window)
    {
      _mpi_data_type = infer_mpi_type( _update_data.data() );
    }

    /**
     * Current size of the bin (number of update elems not yet applied)
     */
    inline long
    size()
    {
      return _update_data.size();
    }

    /**
     * Add an elemental update to this bin
     * \param data Elemental update (r.h.s. of +=)
     * \param ij Linear index of element to be updated
     */
    inline void
    append( T data, long ij )
    {
      _update_data.push_back( data );
      _update_ijs.push_back( ij );
    }

    /**
     * "Flush" this bin, by applying the locally cached updates to the target
     * (which are thereafter discarded on this, the initiating side).
     */
    void
    flush()
    {
      int *blocksize, *displacement;
      MPI_Datatype update_type;
 
      // initialize datatype for update indexing
      blocksize    = new int [size()];
      displacement = new int [size()];
      for ( long i = 0; i < size(); i++ ) {
        blocksize[i] = 1;
        displacement[i] = static_cast<int>(_update_ijs[i]);
#ifndef NOCHECK
        // This is one way to determine if long->int suffered from truncation.
        assert( static_cast<long>(displacement[i]) == _update_ijs[i] );
#endif
      }
      MPI_Type_indexed( size(), blocksize, displacement, _mpi_data_type,
                        &update_type );
      MPI_Type_commit( &update_type );

      // acquire the shared lock, apply the update, unlock
#ifndef USE_MPI_LOCK_ALL
# ifdef USE_MPI_LOCK_SHARED
      MPI_Win_lock( MPI_LOCK_SHARED, _target_rank, LOCK_ASSERT, _target_window );
# else
      MPI_Win_lock( MPI_LOCK_EXCLUSIVE, _target_rank, LOCK_ASSERT, _target_window );
# endif
#endif // USE_MPI_LOCK_ALL
      MPI_Accumulate( _update_data.data(), size(), _mpi_data_type,
                      _target_rank, 0, 1, update_type, MPI_SUM, _target_window );
#ifdef USE_MPI_LOCK_ALL
      // This enforces remote completion.  It may be prudent to use only local
      // completion and enforce remote completion elsewhere using MPI_Win_flush_all.
      MPI_Win_flush( _target_rank, _target_window );
#else // USE_MPI_LOCK_ALL
      MPI_Win_unlock( _target_rank, _target_window );
#endif // USE_MPI_LOCK_ALL

      // clean up after the derived type
      MPI_Type_free( &update_type );
      delete [] blocksize;
      delete [] displacement;

      // clear the local buffers
      clear();
    }

  };

}

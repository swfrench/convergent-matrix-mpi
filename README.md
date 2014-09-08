# Alternative `ConvergentMatrix` based on MPI RMA

An alternative implementation of the
[ConvergentMatrix](http://github.com/swfrench/convergent-matrix)
abstraction, based on one-sided MPI RMA operations.

Namely, the semantics of `MPI_Accumulate` (in combination with `MPI_SUM`)
ensure atomicity of elemental additive augmented-assignment (`+=`) operations.
This is a slightly stronger (finer grained) atomicity constraint than we really
need or use in the UPC++-based version (atomicity of bulk updates), but the
hope is that the MPI runtime will do the "right" (i.e. performant) thing.

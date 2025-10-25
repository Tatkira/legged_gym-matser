#ifndef CSC_H_
#define CSC_H_

typedef struct csc_ {
  OSQPInt nzmax;     ///< maximum number of entries
  OSQPInt m;         ///< number of rows
  OSQPInt n;         ///< number of columns
  OSQPInt* p;        ///< column pointers (size n+1)
  OSQPInt* i;        ///< row indices (size nzmax)
  OSQPFloat* x;      ///< column values (size nzmax)
  OSQPInt nz;        ///< number of entries in matrix
} csc;

#endif /* CSC_H_ */
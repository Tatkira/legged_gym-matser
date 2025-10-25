#ifndef OSQP_DATA_H_
#define OSQP_DATA_H_

typedef struct OSQPData_ {
  OSQPInt n;     ///< Number of variables
  OSQPInt m;     ///< Number of constraints
  OSQPFloat* P;  ///< Cost function matrix (upper triangular)
  OSQPFloat* A;  ///< Constraint matrix
  OSQPFloat* q;  ///< Cost function vector
  OSQPFloat* l;  ///< Lower bound for constraints
  OSQPFloat* u;  ///< Upper bound for constraints
} OSQPData;

#endif /* OSQP_DATA_H_ */
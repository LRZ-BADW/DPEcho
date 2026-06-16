#ifndef _Physics_Common_hpp_
#define _Physics_Common_hpp_

#include "../utils/tb-types.hpp"
#include "../echo.hpp"
#include <sycl/sycl.hpp>

SYCL_EXTERNAL inline void matMul(real m[9], real *vIn, real *vOut, sycl::id<1> gid = 0, unsigned offset = 1){
  const unsigned id0 = gid[0], id1 = id0+offset, id2 = id1+offset;
  vOut[id0] = m[0]*vIn[id0] + m[1] *vIn[id1] + m[2] *vIn[id2];
  vOut[id1] = m[3]*vIn[id0] + m[4] *vIn[id1] + m[5] *vIn[id2];
  vOut[id2] = m[6]*vIn[id0] + m[7] *vIn[id1] + m[8] *vIn[id2];
}

SYCL_EXTERNAL inline real dot(real* u, real* v, unsigned dims = 3, sycl::id<1> gid = 0, unsigned offset = 1){
  real sum = 0.0;
  for(unsigned i=0; i<dims; i++){ sum+=u[gid[0]+i*offset]*v[gid[0]+i*offset]; }
  return sum;
}

SYCL_EXTERNAL inline void cross(real *u, real *v, real *r, sycl::id<1> gid = 0, unsigned offset = 1){
  const unsigned id0 = gid[0], id1 = id0+offset, id2 = id1+offset;
  r[id0] = u[id1]*v[id2] - v[id1]*u[id2];
  r[id1] = u[id2]*v[id0] - v[id2]*u[id0];
  r[id2] = u[id0]*v[id1] - v[id0]*u[id1];
}

#endif

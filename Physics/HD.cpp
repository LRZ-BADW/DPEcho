//   Copyright(C) 2021 Salvatore Cielo, LRZ
//   Copyright(C) 2022 Alexander Pöppl, Intel Corp.
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#include "../Physics.hpp"
#include "../Metric.hpp"

using namespace sycl;

SYCL_EXTERNAL void prim2cons(id<1> myId, unsigned n, real_array v, real_array u, Metric &g){
  const real gDet = g.gDet();
  unsigned const gid = myId[0];
  real vCon[]={v[VX][gid], v[VY][gid], v[VZ][gid]}, vCov[3]; g.con2Cov(vCon, vCov); real v2=dot(vCon,vCov);
  real rh=v[RH][gid], pg=v[PG][gid];
  u[RH][gid]=gDet*rh;         u[PG][gid]=gDet*(0.5*rh*v2+GAMMA1*pg);
  u[VX][gid]=gDet*rh*vCov[0]; u[VY][gid]=gDet*rh*vCov[1]; u[VZ][gid]=gDet*rh*vCov[2];
}

SYCL_EXTERNAL void cons2prim(id<1> myId, unsigned n, real_array u, real_array v, Metric &g, real tol){
  unsigned const gid = myId[0];
  real const gDet1 = g.gDet1();
  real sCov[3]={u[VX][gid]*gDet1,u[VY][gid]*gDet1,u[VZ][gid]*gDet1}, sCon[3]; g.cov2Con(sCov, sCon); real s2=dot(sCov, sCon);
  real rh = u[RH][gid]*gDet1, et=u[PG][gid]*gDet1, rh1=1./rh;
  real pg = sycl::max((GAMMA-1.0)*(et-0.5*rh1*s2), (real)PGFLOOR);
  v[RH][gid] = rh;          v[PG][gid] = pg;
  v[VX][gid] = sCon[0]*rh1; v[VY][gid] = sCon[1]*rh1;  v[VZ][gid] = sCon[2]*rh1;
}

SYCL_EXTERNAL void physicalFlux(int dir, Metric &g, real vD[FLD_TOT], real uD[FLD_TOT], real f[FLD_TOT], real vf[2], real vt[2]){
  real betai[3], gCov[9], gCon[9];     g.g3DCov(gCov); g.beta(betai);
  real const alpha = g.alpha(), gDet = g.g3DCon(gCon);

  const short k1 = (dir+1)-1, k2 = ((dir+1)%3)-1, k3 = ((dir+2)%3)-2;

  real vCov[3], vCon[3] = {vD[VX], vD[VY], vD[VZ]}; matMul(gCov, vCon, vCov);
  const real v2 = dot(vCon,vCov), rh = vD[RH], pg = vD[PG];
  const real wt = .5*rh*v2+GAMMA1*pg;

  uD[RH] = rh     ;  uD[VX] = rh *vCov[0];  uD[VY] = rh *vCov[1];  uD[VZ] = rh *vCov[2];
  uD[PG] = wt - pg;
  for(unsigned iVar = 0; iVar< FLD_TOT; ++iVar){  uD[iVar] *= gDet;}

  //-- Fluxes
  f[VX]= rh*vCon[dir]*vCov[0]; f[VY]= rh*vCon[dir]*vCov[1]; f[VZ]= rh*vCon[dir]*vCov[2];
  f[VX+k1] += pg;
  f[PG] = wt * vCon[dir];       f[RH] = rh * vCon[dir];
  for(unsigned iVar = 0; iVar< FLD_TOT; ++iVar){  f[iVar] *= gDet;}

  //-- Sound speeds
  const real c2 = GAMMA *pg;
  const real vfd = sycl::sqrt( c2 / rh );
  vf[0] = vCon[dir]+vfd;  vf[1] = vCon[dir]-vfd;

  //-- Transverse speeds
  vt[0] = vCon[1+k2];     vt[1] = vCon[2+k3];
}

SYCL_EXTERNAL void physicalSource(id<1> myId, real_array v, Metric &g, real src[4]){
#if METRIC > CARTESIAN
  for(short unsigned i=0; i<4; i++){ src[i] = 0; }; return;

#else
  real ssCon[3][3];
  real betai[3], gCov[9], gCon[9]; g.g3DCov(gCov); g.beta(betai);
  real const alpha=g.alpha(), gDet=g.g3DCon(gCon), gDet1=g.gDet1();
  const real rh=v[RH][myId], pg=v[PG][myId];
  real vCov[3], vCon[]={v[VX][myId],v[VY][myId],v[VZ][myId]}; matMul(gCov, vCon, vCov);
  const real v2 = dot(vCon,vCov);
  const real et=rh;

  real sCon[3] = {rh*vCon[0], rh*vCon[1], rh*vCon[2]};
  for(short unsigned i=0; i<3; i++)
    for(short unsigned j=0; j<3; j++)
      ssCon[i][j] = rh*vCon[i]*vCon[j]+pg*gCon[i*3+j];

  real sum1[3] = {0.0, 0.0, 0.0}, sum2 = 0.0;
  for(short unsigned i=0; i<3; i++)
    for(short unsigned j=0; j<3; j++)
      for(short unsigned iS=0; iS<3; iS++){
        sum1[iS]+=          0.5*ssCon[i ][j] *g.dgCov(i,j,iS);
        sum2    += gCov[i*3+iS]*ssCon[iS][j] *g.dgBeta(i,j);
      }
  real sCov[3]; matMul(gCov,sCon,sCov);
  for(short unsigned iS=0; iS<3; iS++){
    src[iS] = g.alpha()*sum1[iS] - et*g.dgAlpha(iS);
    for(short unsigned k=0; k<3; k++){ src[iS]+= sCov[k] * g.dgBeta(k,iS); }
  }
  src[3] = sum2;
  for(short unsigned k=0; k<3; k++){ src[3]+= betai[k]*sum1[k]-sCon[k]*g.dgAlpha(k); }

  for(short unsigned iS=0; iS<4; iS++){ src[iS]*= gDet; }
  return;

#endif // METRIC type
}

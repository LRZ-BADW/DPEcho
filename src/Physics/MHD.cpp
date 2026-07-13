//  Copyright(C) 2021 Salvatore Cielo, LRZ
//  Copyright(C) 2022 Alexander Pöppl, Intel Corp.
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
  real bCon[]={v[BX][gid], v[BY][gid], v[BZ][gid]}, bCov[3]; g.con2Cov(bCon, bCov); real b2=dot(bCon,bCov);

  real rh=v[RH][gid], pg=v[PG][gid], pt=pg+0.5*b2, wt=0.5*rh*v2+GAMMA1*pg+b2;

  u[RH][gid]=gDet*rh;         u[PG][gid]=gDet*(wt-pt);
  u[VX][gid]=gDet*rh*vCov[0]; u[VY][gid]=gDet*rh*vCov[1]; u[VZ][gid]=gDet*rh*vCov[2];
  u[BX][gid]=gDet   *bCov[0]; u[BY][gid]=gDet   *bCov[1]; u[BZ][gid]=gDet   *bCov[2];
}

SYCL_EXTERNAL void cons2prim(id<1> myId, unsigned n, real_array u, real_array v, Metric &g, real tol){
  unsigned const gid = myId[0];
  real const gDet1 = g.gDet1();
  real sCov[3]={u[VX][gid]*gDet1,u[VY][gid]*gDet1,u[VZ][gid]*gDet1},  sCon[3];  g.cov2Con(sCov, sCon);  real s2=dot(sCov, sCon);
  real bCon[3]={u[BX][gid]*gDet1,u[BY][gid]*gDet1,u[BZ][gid]*gDet1},  bCov[3];  g.con2Cov(bCon, bCov);  real b2=dot(bCov, bCon);

  real rh = u[RH][gid]*gDet1, et=u[PG][gid]*gDet1, rh1=1./rh, pg = sycl::max((GAMMA-1.0)*(et-.5*(rh1*s2+b2)), (real)PGFLOOR);
  v[RH][gid] = rh;          v[PG][gid] = pg;
  v[VX][gid] = sCon[0]*rh1; v[VY][gid] = sCon[1]*rh1;  v[VZ][gid] = sCon[2]*rh1;
  v[BX][gid] = bCon[0]    ; v[BY][gid] = bCon[1]    ;  v[BZ][gid] = bCon[2]    ;
}

SYCL_EXTERNAL void physicalFlux(int dir, Metric &g, real vD[FLD_TOT], real uD[FLD_TOT], real f[FLD_TOT], real vf[2], real vt[2]){
  real betai[3], gCov[9], gCon[9];     g.g3DCov(gCov); g.beta(betai);
  real const alpha = g.alpha(), gDet = g.g3DCon(gCon);

  const short k1 = (dir+1)-1, k2 = ((dir+1)%3)-1, k3 = ((dir+2)%3)-2;

  //-- Assignments
  real vCov[3], vCon[3] = {vD[VX], vD[VY], vD[VZ]}; matMul(gCov, vCon, vCov);
  real bCov[3], bCon[3] = {vD[BX], vD[BY], vD[BZ]}; matMul(gCov, bCon, bCov);
  const real v2 = dot(vCon,vCov),  b2 = dot(bCon,bCov),  vb = dot(vCov,bCon);
  const real rh = vD[RH],  pg = vD[PG],  pt = pg+.5*b2,  wt =.5*rh*v2+GAMMA1*pg+b2;

  //-- Conserved
  uD[RH] = rh     ;  uD[VX] = rh *vCov[0];  uD[VY] = rh *vCov[1];  uD[VZ] = rh *vCov[2];
  uD[PG] = wt - pt;  uD[BX] =      vD[BX];  uD[BY] =      vD[BY];  uD[BZ] =      vD[BZ];
  for(unsigned iVar = 0; iVar< FLD_TOT; ++iVar){  uD[iVar] *= gDet;}

  //-- Fluxes
  f[VX]= rh*vCon[dir]*vCov[0] - bCon[dir]*bCov[0]; f[BX]= vCon[dir]*bCon[0]-bCon[dir]*vCon[0];
  f[VY]= rh*vCon[dir]*vCov[1] - bCon[dir]*bCov[1]; f[BY]= vCon[dir]*bCon[1]-bCon[dir]*vCon[1];
  f[VZ]= rh*vCon[dir]*vCov[2] - bCon[dir]*bCov[2]; f[BZ]= vCon[dir]*bCon[2]-bCon[dir]*vCon[2];
  f[VX+k1] += pt;
  f[PG] = wt*vD[VX+dir] - vb*vD[BX+dir];           f[RH] = rh * vD[BX+dir];
  for(unsigned iVar = 0; iVar< FLD_TOT; ++iVar){  f[iVar] *= gDet;}

  //-- Fast magnetosonic speeds (vCon along direction dir)
  const real c2 = GAMMA *pg, a2 = c2+b2;
  const real comfort = gCon[dir+3*dir]*a2*a2 - 4.0*c2*bCon[dir]*bCon[dir];
  const real vfd = sycl::sqrt( 0.5*( a2+sycl::sqrt( sycl::max((real)0.0,(real)comfort) ) )/rh );
  vf[0] = vCon[dir]+vfd;  vf[1] = vCon[dir]-vfd;

  //-- Transverse speeds
  vt[0] = vCon[1+k2];     vt[1] = vCon[2+k3];
}

SYCL_EXTERNAL void physicalSource(id<1> myId, real_array v, Metric &g, real src[4]){
#if METRIC > CARTESIAN
  for(short unsigned i=0; i<4; i++){ src[i] = 0; }; return;

#else
  real ssCon[3][3];
  //-- Metric components
  real betai[3], gCov[9], gCon[9]; g.g3DCov(gCov); g.beta(betai);
  real const alpha=g.alpha(), gDet=g.g3DCon(gCon), gDet1=g.gDet1();
  //-- Readouts (all cell-centered values, unlike in physicalFlux)
  const real rh=v[RH][myId], pg=v[PG][myId];
  real bCov[3], bCon[]={v[BX][myId],v[BY][myId],v[BZ][myId]};
  matMul(gCov, bCon, bCov); const real b2 = dot(bCon,bCov);

  const real pt=pg+.5*b2, et=rh, vCon[]={v[VX][myId],v[VY][myId],v[VZ][myId]};
  real sCon[3] = {rh*vCon[0], rh*vCon[1], rh*vCon[2]};
  for(short unsigned i=0; i<3; i++)
    for(short unsigned j=0; j<3; j++)
      ssCon[i][j] = rh*vCon[i]*vCon[j]-bCon[i]*bCon[j]+pt*gCon[i*3+j];

  // WARNING: source tested only for CARTESIAN case (trivial!)
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

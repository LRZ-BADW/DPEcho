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
  real rh=v[RH][gid], pg=v[PG][gid];

  real vCon[]={v[VX][gid], v[VY][gid], v[VZ][gid]}, vCov[3]; g.con2Cov(vCon, vCov); real u2=dot(vCon,vCov); real glf=sycl::sqrt(1.0+u2);
  vCov[0] *= 1.0/glf; vCov[1] *= 1.0/glf; vCov[2] *= 1.0/glf;
  vCon[0] *= 1.0/glf; vCon[1] *= 1.0/glf; vCon[2] *= 1.0/glf;

  real d=rh*glf, h=rh+GAMMA1*pg, w=h*glf*glf, w1=d*u2/(1.+glf)+GAMMA1*pg*glf*glf;

  u[RH][gid] = d; u[PG][gid] = w1-pg;
  u[VX][gid] = w*vCov[0]; u[VY][gid] = w*vCov[1]; u[VZ][gid] = w*vCov[2];

  for(unsigned short iFld=0; iFld<FLD_TOT; ++iFld){ u[iFld][gid] *= gDet; }
}

SYCL_EXTERNAL void cons2prim(id<1> myId, unsigned n, real_array u, real_array v, Metric &g, real tol){
  unsigned const gid = myId[0];
  real const gDet1 = g.gDet1();
  real sCov[3]={u[VX][gid]*gDet1,u[VY][gid]*gDet1,u[VZ][gid]*gDet1},  sCon[3];  g.cov2Con(sCov, sCon);  real s2=dot(sCov, sCon);

  real d=u[RH][gid]*gDet1, et1=u[PG][gid]*gDet1, w1, u2, glf;

  real com = (et1+d); w1 = 4.*com*com - 3.*s2;
  w1 = sycl::max( (2.*com+sycl::sqrt(sycl::max(w1,(real)0.)))/3.-d, (real)0.);

  real w, pg, fw, dv2, dpg, dfw, dw1;
  for(unsigned iter=0; iter<20; ++iter){
    w  = w1+d;  u2 = sycl::max(s2/(w*w-s2), (real)0.0); glf = sycl::sqrt(1.+u2);
    pg = (w1-d*u2/(1.+glf))/(GAMMA1*glf*glf); fw = w1-et1-pg;
    dv2 = -2.*s2/(w*w*w);
    dpg = 1./(GAMMA1*glf*glf) - glf*(0.5*d/GAMMA1+glf*pg)*dv2;  dfw = 1.-dpg;  dw1 = -fw/dfw;
    if (sycl::fabs(dw1) < tol*w1){ break; }else{ w1+= dw1;}
  }

  real rh = d/glf; pg = sycl::max(pg, (real)PGFLOOR);
  real vCon[3]={ sCon[0]/w, sCon[1]/w, sCon[2]/w};

  bool bad = (glf != glf) || (glf < (real)1.0) || (w != w) || (w <= (real)0);
  if (bad) {
    glf = (real)1.0; rh = d; pg = (real)PGFLOOR;
    for (int i=0; i<3; ++i) vCon[i] = (sCon[i]==sCon[i] && d>(real)0) ? sCon[i]/d : (real)0;
  }

  v[RH][gid] = rh;          v[PG][gid] = pg;
  v[VX][gid] = vCon[0]*glf; v[VY][gid] = vCon[1]*glf;  v[VZ][gid] = vCon[2]*glf;
}

SYCL_EXTERNAL void physicalFlux(int dir, Metric &g, real vD[FLD_TOT], real uD[FLD_TOT], real f[FLD_TOT], real vf[2], real vt[2]){
  real betai[3], gCov[9], gCon[9];     g.g3DCov(gCov); g.beta(betai);
  real const alpha = g.alpha(), gDet = g.g3DCon(gCon);

  const short k1 = (dir+1)-1, k2 = ((dir+1)%3)-1, k3 = ((dir+2)%3)-2;

  real rh = vD[RH], pg = vD[PG], vCon[] = {vD[VX], vD[VY], vD[VZ]};
  real vCov[3];                    matMul(gCov, vCon, vCov);
  const real u2 = dot(vCov,vCon),  glfInv = sycl::rsqrt(1+u2), glf = 1.0/glfInv;
  vCon[0] *= glfInv;  vCon[1] *= glfInv;  vCon[2] *= glfInv;
  vCov[0] *= glfInv;  vCov[1] *= glfInv;  vCov[2] *= glfInv;

  const real d = rh/glfInv,  h = rh + GAMMA1*pg,  w = h * glf * glf;
  const real w1= d*u2/(1.+glf) + GAMMA1*pg*glf*glf;

  uD[RH] = d;   uD[PG] = w1-pg;
  uD[VX] = w*vCov[0]; uD[VY] = w*vCov[1]; uD[VZ] = w*vCov[2];
  for(unsigned iVar = 0; iVar< FLD_TOT; ++iVar){ uD[iVar]*= gDet;}

  f[VX] = w*vCon[dir]*vCov[0];
  f[VY] = w*vCon[dir]*vCov[1];
  f[VZ] = w*vCon[dir]*vCov[2];
  f[VX+k1]+= pg;
  f[RH] = d*vCon[dir];    f[PG] = w1*vCon[dir];

  f[RH] = gDet*alpha*f[RH] - betai[dir]*uD[RH];
  f[VX] = gDet*alpha*f[VX] - betai[dir]*uD[VX];
  f[VY] = gDet*alpha*f[VY] - betai[dir]*uD[VY];
  f[VZ] = gDet*alpha*f[VZ] - betai[dir]*uD[VZ];
  f[PG] = gDet*alpha*f[PG] - betai[dir]*uD[PG];

  const real cs2=GAMMA*pg/h, v2 = u2/(1.0+u2);
  const real vf1 = vCon[dir]*(1.0-cs2)/(1.0-v2*cs2);
  const real vf2 = sycl::sqrt( cs2*glfInv*glfInv* ( (1.-v2*cs2)*gCon[4*dir]-(1.-cs2)*vCon[dir]*vCon[dir])) / (1.-v2*cs2);
  vf[0] = alpha*(vf1+vf2)-betai[dir];
  vf[1] = alpha*(vf1-vf2)-betai[dir];

  vt[0] = alpha*vCon[1+k2]-betai[1+k2];
  vt[1] = alpha*vCon[2+k3]-betai[2+k3];
}

SYCL_EXTERNAL void physicalSource(id<1> myId, real_array v, Metric &g, real src[4]){
  real ssCon[3][3];
  real betai[3], gCov[9], gCon[9]; g.g3DCov(gCov); g.beta(betai);
  real const alpha=g.alpha(), gDet=g.g3DCon(gCon), gDet1=g.gDet1();
  const real rh=v[RH][myId], pg=v[PG][myId];

  real vCov[3], vCon[]={v[VX][myId],v[VY][myId],v[VZ][myId]}; matMul(gCov, vCon, vCov);
  const real u2 = dot(vCov,vCon), glfInv = sycl::rsqrt(1+u2), glf = 1.0/glfInv;
  vCon[0] *= glfInv; vCon[1] *= glfInv; vCon[2] *= glfInv;
  vCov[0] *= glfInv; vCov[1] *= glfInv; vCov[2] *= glfInv;

  const real d=rh/glfInv, h=rh+GAMMA1*pg, w=h*glf*glf, pt=pg, et=w-pg;

  real sCon[3] = { w*vCon[0], w*vCon[1], w*vCon[2] };

  for(short unsigned i=0; i<3; i++)
    for(short unsigned j=0; j<3; j++)
      ssCon[i][j] = w*vCon[i]*vCon[j]+pt*g.gCon(i,j);

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
}

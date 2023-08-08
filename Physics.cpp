//  Copyright(C) 2021 Salvatore Cielo, LRZ
//  Copyright(C) 2022 Alexander Pöppl, Intel Corp.
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#include "Physics.hpp"
#include "Metric.hpp"

//-- Manual 3-vectors operation in GR
SYCL_EXTERNAL void matMul(field m[9], field *vIn, field *vOut,  id<1> gid, unsigned offset){
  const unsigned id0 = gid[0], id1 = id0+offset, id2 = id1+offset;
  vOut[id0] = m[0]*vIn[id0] + m[1] *vIn[id1] + m[2] *vIn[id2];
  vOut[id1] = m[3]*vIn[id0] + m[4] *vIn[id1] + m[5] *vIn[id2];
  vOut[id2] = m[6]*vIn[id0] + m[7] *vIn[id1] + m[8] *vIn[id2];
} // Multiplication by STATIC 3x3 matrix. Sorry it can't be inlined

SYCL_EXTERNAL field dot(field* u, field* v, unsigned dims, id<1> gid, unsigned offset){
  field sum = 0.0;
  for(unsigned i=0; i<dims; i++){ sum+=u[gid[0]+i*offset]*v[gid[0]+i*offset]; }
  return sum;
} // Sorry it can't be inlined. Otherwise it should go straigth to MKL

SYCL_EXTERNAL void  cross(field *u, field *v, field *r   , id<1> gid, unsigned offset){
  const unsigned id0 = gid[0], id1 = id0+offset, id2 = id1+offset;
  r[id0] = u[id1]*v[id2] - v[id1]*u[id2];
  r[id1] = u[id2]*v[id0] - v[id2]*u[id0];
  r[id2] = u[id0]*v[id1] - v[id0]*u[id1];
}

//-- Conversion of quantities
SYCL_EXTERNAL void prim2cons(id<1> myId, unsigned n, field_array v, field_array u, Metric &g){
 const field gDet = g.gDet();
 unsigned const gid = myId[0];
#if PHYSICS==MHD
  field vCon[]={v[VX][gid], v[VY][gid], v[VZ][gid]}, vCov[3]; g.con2Cov(vCon, vCov); field v2=dot(vCon,vCov);
  field bCon[]={v[BX][gid], v[BY][gid], v[BZ][gid]}, bCov[3]; g.con2Cov(bCon, bCov); field b2=dot(bCon,bCov);

  field rh=v[RH][gid], pg=v[PG][gid], pt=pg+0.5*b2, wt=0.5*rh*v2+GAMMA1*pg+b2;

  u[RH][gid]=gDet*rh;         u[PG][gid]=gDet*(wt-pt);
  u[VX][gid]=gDet*rh*vCov[0]; u[VY][gid]=gDet*rh*vCov[1]; u[VZ][gid]=gDet*rh*vCov[2];
  u[BX][gid]=gDet   *bCov[0]; u[BY][gid]=gDet   *bCov[1]; u[BZ][gid]=gDet   *bCov[2];

#elif PHYSICS==GRMHD
  field rh=v[RH][gid], pg=v[PG][gid];

  field vCon[]={v[VX][gid], v[VY][gid], v[VZ][gid]}, vCov[3]; g.con2Cov(vCon, vCov); field u2=dot(vCon,vCov); field glf=mysycl::sqrt(1.0+u2);
  field bCon[]={v[BX][gid], v[BY][gid], v[BZ][gid]}, bCov[3]; g.con2Cov(bCon, bCov); field b2=dot(bCon,bCov);
  vCov[0] *= 1.0/glf; vCov[1] *= 1.0/glf; vCov[2] *= 1.0/glf;
  vCon[0] *= 1.0/glf; vCon[1] *= 1.0/glf; vCon[2] *= 1.0/glf;

  field eCov[3]; cross(vCon, bCon, eCov); eCov[0]*=-gDet; eCov[1]*=-gDet; eCov[2]*=-gDet;
  field eCon[3]; g.cov2Con(eCov, eCon);  field e2=dot(eCov,eCon), uem=0.5*(e2+b2);
  field sCov[3]; cross(eCon, bCon, sCov); sCov[0]*= gDet; sCov[1]*= gDet; sCov[2]*= gDet;
  field sCon[3]; g.cov2Con(sCov, sCon);

  field d=rh*glf, h=rh+GAMMA1*pg, w=h*glf*glf, w1=d*u2/(1.+glf)+GAMMA1*pg*glf*glf; // W'=W-D

  u[RH][gid] = d; u[PG][gid] = w1-pg+uem;
  u[VX][gid] = w*vCov[0]+sCov[0]; u[VY][gid] = w*vCov[1]+sCov[1]; u[VZ][gid] = w*vCov[2]+sCov[2];
  u[BX][gid] =   bCon[0]        ; u[BY][gid] =   bCon[1]        ; u[BZ][gid] =   bCon[2]        ;

  for(unsigned short iFld=0; iFld<FLD_TOT; ++iFld){ u[iFld][gid] *= gDet; }

#endif // PHYSICS
}

SYCL_EXTERNAL void cons2prim(id<1> myId, unsigned n, field_array u, field_array v, Metric &g){
  unsigned const gid = myId[0];
  field const gDet1 = 1./g.gDet();
  field sCov[3]={u[VX][gid]*gDet1,u[VY][gid]*gDet1,u[VZ][gid]*gDet1},  sCon[3];  g.cov2Con(sCov, sCon);  field s2=dot(sCov, sCon);
  field bCon[3]={u[BX][gid]*gDet1,u[BY][gid]*gDet1,u[BZ][gid]*gDet1},  bCov[3];  g.con2Cov(bCon, bCov);  field b2=dot(bCov, bCon);

#if PHYSICS==MHD
  field rh = u[RH][gid]*gDet1,  et=u[PG][gid]*gDet1,   rh1=1./rh,   pg = (GAMMA-1.0)*(et-.5*(rh1*s2+b2));
  v[RH][gid] = rh;          v[PG][gid] = pg;
  v[VX][gid] = sCon[0]*rh1; v[VY][gid] = sCon[1]*rh1;  v[VZ][gid] = sCon[2]*rh1;
  v[BX][gid] = bCon[0]    ; v[BY][gid] = bCon[1]    ;  v[BZ][gid] = bCon[2]    ;

#elif PHYSICS==GRMHD
  field sb=dot(sCov,bCon),  sb2=sb*sb,  b2st2=mysycl::max(s2*b2-sb2, 0.);
  field d=u[RH][gid]*gDet1, et1=u[PG][gid]*gDet1, w1, u2, glf;

  // The folowing two are alternative: Initial guess from old values ...
  //field uCon[] = {v[VX][gid], v[VY][gid], v[VZ][gid]},  uCov[3];  g.con2Cov(uCon, uCov);  field u2=dot(uCov,uCon);  glf=mysycl::sqrt(1.+u2);
  //w1 = d*u2/(1.0+glf)+GAMMA1*v[PG][gid]*glf*glf;
  // ... OR from quadratic equation (TODO: comment either or put a CMake switch)
  field com = (et1+d-b2);   w1 = 4.*com*com - 3.*(s2-b2*(2.*et1+d-b2));
  w1 = mysycl::max( (2.*com+mysycl::sqrt(mysycl::max(w1,0.)))/3.-d,  0.);

  //-- Undetermined iteration. TODO: is there a SYCL tool for such cases?  (ALL LOCAL, luckily).
  field w, vv2, pg, fw, dv2, dpg, dfw, dw1;  const field tol=1.e-9;
  for(unsigned iter=0; iter<20; ++iter){
    w  = w1+d;  vv2 = w*w*s2+(2.*w+b2)*sb2;   com = w*(w+b2);  u2 = vv2/(com*com-vv2); glf = mysycl::sqrt(1.+u2);
    pg = (w1-d*u2/(1.+glf))/(GAMMA1*glf*glf); com = w+b2;      fw = w1-et1-pg + .5*(b2 + b2st2/(com*com));
    dv2 = -2.*( s2+sb2* (3.*w*(com)+b2*b2)/(w*w*w) )/ (com*com*com);
    dpg = 1./(GAMMA1*glf*glf) - glf*(.5*d/GAMMA1+glf*pg)*dv2;  dfw = 1.-dpg-b2st2/(com*com*com);  dw1 = -fw/dfw;
    if (abs(dw1) < tol*w1){ break; }else{ w1+= dw1;}
  }

  field rh = d/glf;  pg = mysycl::max(pg, (field)PGFLOOR);
  field vCon[3]={ (sCon[0]+sb*bCon[0]/w)/(w+b2), (sCon[1]+sb*bCon[1]/w)/(w+b2), (sCon[2]+sb*bCon[2]/w)/(w+b2)};

  v[RH][gid] = rh;          v[PG][gid] = pg;
  v[VX][gid] = vCon[0]*glf; v[VY][gid] = vCon[1]*glf;  v[VZ][gid] = vCon[2]*glf;
  v[BX][gid] = bCon[0]    ; v[BY][gid] = bCon[1]    ;  v[BZ][gid] = bCon[2]    ;

#endif // PHYSICS==GRMHD

}

//-- Fluxes and characteristic velocities.
//    IMPORTANT: all local quantities, they have been sampled --> access simply by eg. f[VX]
SYCL_EXTERNAL void physicalFlux(int dir, Metric &g, field vD[FLD_TOT], field uD[FLD_TOT], field f[FLD_TOT], field vf[2], field vt[2] ){
  field alpha = g.alpha(), betai[3], gCov[9], gCon[9];
  g.beta(betai);
  g.g3DCov(gCov);
  field const gDet =  g.g3DCon(gCon);

  const short k1 = (dir+1)-1, k2 = ((dir+1)%3)-1, k3 = ((dir+2)%3)-2;

#if PHYSICS==MHD
  //-- Assignments
  field vCov[3], vCon[3] = {vD[VX], vD[VY], vD[VZ]}; matMul(gCov, vCon, vCov);
  field bCov[3], bCon[3] = {vD[BX], vD[BY], vD[BZ]}; matMul(gCov, bCon, bCov);
  const field v2 = dot(vCon,vCov),  b2 = dot(bCon,bCov),  vb = dot(vCov,bCon);
  const field rh = vD[RH],  pg = vD[PG],  pt = pg+.5*b2,  wt =.5*rh*v2+GAMMA1*pg+b2;

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
  const field c2 = GAMMA *pg, a2 = c2+b2;
  const field comfort = gCon[dir+3*dir]*a2*a2 - 4.0*c2*bCon[dir]*bCon[dir];
  const field vfd = mysycl::sqrt( 0.5*( a2+mysycl::sqrt( mysycl::max((field)0.0,(field)comfort) ) )/rh );
  vf[0] = vCon[dir]+vfd;  vf[1] = vCon[dir]-vfd;

  //-- Transverse speeds
  vt[0] = vCon[1+k2];     vt[1] = vCon[2+k3];

#elif PHYSICS==GRMHD

  //-- Assignments
  field rh = vD[RH], pg = vD[PG], vCon[] = {vD[VX], vD[VY], vD[VZ]}, bCon[] = {vD[BX], vD[BY], vD[BZ]};
  field vCov[3], bCov[3];         matMul(gCov, vCon, vCov);  matMul(gCov, bCon,  bCov);
  const field u2 = dot(vCov,vCon), b2 = dot(bCon,bCov),  glfInv = mysycl::rsqrt(1+u2), glf = 1.0/glfInv;
  vCon[0] *= glfInv;  vCon[1] *= glfInv;  vCon[2] *= glfInv;
  vCov[0] *= glfInv;  vCov[1] *= glfInv;  vCov[2] *= glfInv;

  field eCov[3]; cross(vCon, bCon, eCov);  eCov[0]*=-gDet;  eCov[1]*=-gDet;  eCov[2]*=-gDet;
  field eCon[3]; matMul(gCon, eCov, eCon);  const field e2 = dot(eCov,eCon);
  field sCov[3]; cross(eCon, bCon, sCov);  sCov[0]*= gDet;  sCov[1]*= gDet;  sCov[2]*= gDet;
  field sCon[3]; matMul(gCon, sCov, sCon);

  const field d = rh/glfInv,  h = rh + GAMMA1*pg,  w = h * glf * glf,   uem = 0.5*(e2+b2);
  const field w1= d*u2/(1.+glf) + GAMMA1*pg*glf*glf; // W' = W-D

  //-- Conserved
  uD[RH] = d;   uD[PG] = w1-pg+uem;
  uD[VX] = w*vCov[0]+sCov[0]; uD[VY] = w*vCov[1]+sCov[1]; uD[VZ] = w*vCov[2]+sCov[2];
  uD[BX] =   bCon[0]        ; uD[BY] =   bCon[1];         uD[BZ] =   bCon[2];
  for(unsigned iVar = 0; iVar< FLD_TOT; ++iVar){ uD[iVar]*= gDet;}

  //-- Fluxes
  f[VX] = w*vCon[dir]*vCov[0]-eCon[dir]*eCov[0]-bCon[dir]*bCov[0];
  f[VY] = w*vCon[dir]*vCov[1]-eCon[dir]*eCov[1]-bCon[dir]*bCov[1];
  f[VZ] = w*vCon[dir]*vCov[2]-eCon[dir]*eCov[2]-bCon[dir]*bCov[2];
  f[VX+k1]+= pg+uem;
  f[RH] = d*vCon[dir];    f[PG] = w1*vCon[dir]+sCon[dir]; // Seems odd

  f[RH] = gDet*alpha*f[RH] - betai[dir]*uD[RH];
  f[VX] = gDet*alpha*f[VX] - betai[dir]*uD[VX];
  f[VY] = gDet*alpha*f[VY] - betai[dir]*uD[VY];
  f[VZ] = gDet*alpha*f[VZ] - betai[dir]*uD[VZ];
  f[PG] = gDet*alpha*f[PG] - betai[dir]*uD[PG];

  field tmp[3];  cross(betai, bCon, tmp);
  eCov[0] = alpha*eCov[0] + gDet * tmp[0];
  eCov[1] = alpha*eCov[1] + gDet * tmp[1];
  eCov[2] = alpha*eCov[2] + gDet * tmp[2];

  f[BX+k1] = 0.0;      f[BY+k2] =-eCov[2+k3];    f[BZ+k3] = eCov[1+k2];

  //-- Fast magnetosonic speeds (vCon along direction dir)
  const field cs2=GAMMA*pg/h, ca2=1.0-h/(h+mysycl::max(b2-e2,0.)),  a2=cs2+ca2-cs2*ca2,  v2 = u2/(1.0+u2);
  const field vf1 = vCon[dir]*(1.0-a2)/(1.0-v2*a2);
  const field vf2 = mysycl::sqrt( a2*glfInv*glfInv* ( (1.-v2*a2)*gCon[4*dir]-(1.-a2)*vCon[dir]*vCon[dir]))/(1.-v2*a2);
  vf[0] = alpha*(vf1+vf2)-betai[dir];
  vf[1] = alpha*(vf1-vf2)-betai[dir];

  //-- Transverse speeds (vCon)
  vt[0] = alpha*vCon[1+k2]-betai[1+k2];
  vt[1] = alpha*vCon[2+k3]-betai[2+k3];

#endif // PHYSICS

}

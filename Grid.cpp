//   Copyright(C) 2020 Fabio Baruffa, Intel Corp.
//   Copyright(C) 2021 Salvatore Cielo, LRZ
//   Copyright(C) 2021 Alexander Pöppl, Intel Corp.
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#include "Grid.hpp"
#include "Logger.hpp"

//-- Constructor; better all in one go
Grid::Grid( int nx, int ny, int nz, int hx, int hy, int hz, field xmin, field xmax,field ymin, field ymax, field zmin, field zmax):
  n  {nx , ny , nz},  h{hx, hy, hz},  nh  { nx+2*hx, ny+2*hy, nz+2*hz},
  nt {nx * ny * nz},  nht{nh[0] * nh[1] * nh[2]},
  xMin{ xmin,ymin,zmin},               xMax{xmax,ymax,zmax},
  dx  {(xmax-xmin)/((field)nx), (ymax-ymin)/((field)ny), (zmax-zmin)/((field)nz)},
  hMin{ xmin-hx*dx[0], ymin-hy*dx[1], zmin-hz*dx[2]},
  hMax{ xmax+hx*dx[0], ymax+hy*dx[1], zmax+hz*dx[2]}{
  // This is a memorial for a nice algo :)
  /*
  while(gMax%wgMax){wgMax--;}
  groupSize[2] = std::gcd(gMax                          , n[2]); groupSize[2] = std::min(groupSize[2],wgMax);
  groupSize[1] = std::gcd(gMax/groupSize[2]             , n[1]); groupSize[1] = std::min(groupSize[1],wgMax);
  groupSize[0] = std::gcd(gMax/groupSize[2]/groupSize[1], n[0]); groupSize[0] = std::min(groupSize[0],wgMax);
  */
}

//-- Infos
void Grid::print() {
  Log::cout(4) << TAG
    << " halos_("  << h [0] << " " << h [1] << " " << h [2]<< ")  "
    << " cellsWH(" << nh[0] << " " << nh[1] << " " << nh[2]<< ")  "
    << " cellsNH(" << n [0] << " " << n [1] << " " << n [2]<< ")  "
    << " Dxyz("    << dx[0] << " " << dx[1] << " " << dx[2]<< ")  "
    << Log::endl;
}

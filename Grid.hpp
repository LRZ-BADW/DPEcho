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

#ifndef _Grid_hpp_
#define _Grid_hpp_

#include "utils/tb-types.hpp"
#include "echo.hpp"

#include <sycl/sycl.hpp>
#include <numeric>

class Grid {

  public: // All numbers of points along the 3 directions
    const int n[3], h[3], nh[3], nt, nht;
    const field xMin[3], xMax[3], dx[3], hMin[3], hMax[3];  // Physical point distances

    Grid(int, int, int, int, int, int, field, field, field, field, field, field);

    void print();

    //-- Device code: beware the indexing!
    // Retrieve cell Coordinates (USE INNER grid indexing, a.k.a NH indexing!!)
    // If it the parallel_for range is NOT NH, add proper offsets to global_id() to get i3
    inline field xL (     sycl::id<3> i3, int myDir) const { return xMin[myDir] + dx[myDir] * i3[myDir]     ; }
    inline field xC (     sycl::id<3> i3, int myDir) const { return xMin[myDir] + dx[myDir] *(i3[myDir]+0.5); }
    inline field xR (     sycl::id<3> i3, int myDir) const { return xMin[myDir] + dx[myDir] *(i3[myDir]+1.0); }
    // Convenient shortcuts when parallel_for range is NH, you can just pass the nd_item
    inline field xL (sycl::nd_item<3> it, int myDir) const { return xMin[myDir] + dx[myDir] * it.get_global_id(myDir)     ; }
    inline field xC (sycl::nd_item<3> it, int myDir) const { return xMin[myDir] + dx[myDir] *(it.get_global_id(myDir)+0.5); }
    inline field xR (sycl::nd_item<3> it, int myDir) const { return xMin[myDir] + dx[myDir] *(it.get_global_id(myDir)+1.0); }
    // Same as above, but for full WH indexing (based on hMin values)
    inline field xLh(sycl::nd_item<3> it, int myDir) const { return hMin[myDir] + dx[myDir] * it.get_global_id(myDir)     ; }
    inline field xCh(sycl::nd_item<3> it, int myDir) const { return hMin[myDir] + dx[myDir] *(it.get_global_id(myDir)+0.5); }
    inline field xRh(sycl::nd_item<3> it, int myDir) const { return hMin[myDir] + dx[myDir] *(it.get_global_id(myDir)+1.0); }
};

//-- General Indexing methods (work for any SYCL range, not just the grid!)
// Call in parallel_for. Calculate most general linear index to access arrays
// Offset can be positive or negative. Positve -> target array starts before parallel_for range.

// This version uses sycl::ids for the offset calculation.
inline size_t globLinId(sycl::id<3> const baseId, sycl::range<3> const r, sycl::id<3> const offset) {
  sycl::id<3> id = baseId + offset;
  return id[2] + id[1] * r[2] + id[0] * r[1] * r[2];
}

// this one uses int arrays. They should do the same thing.
inline int globLinId(   sycl::id<3> myId, const int arrRange[3], const int relOffset[3]) {
  myId[0]+= relOffset[0]; myId[1]+= relOffset[1];  myId[2]+= relOffset[2];// Offset to recenter your indexing
  return myId[2] + myId[1]*arrRange[2] + myId[0]*arrRange[1]*arrRange[2]; // From SYCL API. Updating this line updates all.
}

inline int stride(sycl::id<3> id, int myDir, const int arrRange[3]){
  return 1 * !!(myDir == 2) +  arrRange[2] * !!(myDir == 1) + arrRange[1] * arrRange[2] * !!(myDir == 0);
}

inline size_t nextMultiple(size_t x, size_t n) {
  return ((x + n - 1) / n) * n;
}

inline sycl::nd_range<3> getMatchingNdRange(sycl::range<3> const &rGlob, sycl::range<3> const &rLoc) {
  sycl::range<3> rGlobFit(nextMultiple(rGlob[0], rLoc[0]), nextMultiple(rGlob[1], rLoc[1]), nextMultiple(rGlob[2], rLoc[2]));
  return sycl::nd_range<3>(rGlobFit, rLoc);
}

inline bool isOutOfBounds(sycl::id<3> const id, sycl::range<3> const range) {
  return id[0] >= range[0] || id[1] >= range[1] || id[2] >= range[2];
}


#endif

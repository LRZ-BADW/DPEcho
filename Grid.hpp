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

#include "echo.hpp"
#include "Logger.hpp"
#include "utils/tb-types.hpp"

#include "echoSycl.hpp"

/* TODO: Put something like this there!
enum class PositionInCell {
  CENTER, X_BORDER, Y_BORDER, Z_BORDER
};

or this

int const POS_CENTER = 0;
*/

class Grid {

  public:
    const int n[3], h[3], nh[3], nt, nht; // All numbers of points along the 3 directions
    const field xMin[3], xMax[3], dx[3], hMin[3], hMax[3];  // Physical point distances

    Grid(int, int, int, int, int, int, field, field, field, field, field, field);

    void print();

    //-- Device code: beware the indexing!
    // Retrieve cell Coordinates (USE INNER grid indexing, a.k.a NH indexing!!)
    // If it the parallel_for range is NOT NH, add proper offsets to global_id() to get i3
    inline const field xL (     id<3> i3, int myDir) const { return xMin[myDir] + dx[myDir] * i3[myDir]      ; }
    inline const field xC (     id<3> i3, int myDir) const { return xMin[myDir] + dx[myDir] *(i3[myDir]+0.5f); }
    inline const field xR (     id<3> i3, int myDir) const { return xMin[myDir] + dx[myDir] *(i3[myDir]+1.0f); }
    // Convenient shortcuts when parallel_for range is NH, you can just pass the nd_item
    inline const field xL (nd_item<3> it, int myDir) const { return xMin[myDir] + dx[myDir] * it.get_global_id(myDir)      ; }
    inline const field xC (nd_item<3> it, int myDir) const { return xMin[myDir] + dx[myDir] *(it.get_global_id(myDir)+0.5f); }
    inline const field xR (nd_item<3> it, int myDir) const { return xMin[myDir] + dx[myDir] *(it.get_global_id(myDir)+1.0f); }
    // Same as above, but for full WH indexing (based on hMin values)
    inline const field xLh(nd_item<3> it, int myDir) const { return hMin[myDir] + dx[myDir] * it.get_global_id(myDir)      ; }
    inline const field xCh(nd_item<3> it, int myDir) const { return hMin[myDir] + dx[myDir] *(it.get_global_id(myDir)+0.5f); }
    inline const field xRh(nd_item<3> it, int myDir) const { return hMin[myDir] + dx[myDir] *(it.get_global_id(myDir)+1.0f); }

    // put some grid id to position in 3d calculator here
    /*
    if constexpr (pos == PositionInCell::CENTER) {

    } else if constexpr (pos == PositionInCell::X_BORDER) {
    } else if constexpr (pos == PositionInCell::X_BORDER) {
    } else if constexpr (pos == PositionInCell::X_BORDER) {
    } else {
      coords[0] = 0.0;
      coords[1] = 0.0;
      coords[2] = 0.0;
    }*/
};

//-- General Indexing methods (work for any SYCL range, not just the grid!)
// Call in parallel_for. Calculate most general linear index to access arrays
// Offset can be positive or negative. Positve -> target array starts before parallel_for range.
inline int globLinId(   id<3> myId, const int arrRange[3], const int relOffset[3]) {
  myId[0]+= relOffset[0]; myId[1]+= relOffset[1];  myId[2]+= relOffset[2];// Offset to recenter your indexing
  return myId[2] + myId[1]*arrRange[2] + myId[0]*arrRange[1]*arrRange[2]; // From SYCL API. Updating this line updates all.
}

inline int stride(id<3> id, int myDir, const int arrRange[3]){
  int off[] = {0, 0, 0}; auto id0 = globLinId(id, arrRange, off);
  off[myDir]= 1;         auto id1 = globLinId(id, arrRange, off);
  return id1 - id0; // Wasteful but SAFE(r)
}

inline int stride(int myDir, const int arrRange[3]){
  id<3> id(0,0,0);
  return stride(id,myDir,arrRange); // Wasteful but SAFE(r)
}

#endif

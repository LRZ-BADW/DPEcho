//  Copyright(C) 2020 Fabio Baruffa, Intel Corp.
//  Copyright(C) 2022 Salvatore Cielo, LRZ
//  Copyright(C) 2022 Alexander Pöppl, Intel Corp.
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#ifndef _TOOLBOX_TIMER_H_
#define _TOOLBOX_TIMER_H_

#include <string>
#ifdef MPICODE
#include <mpi.h>
#else
#include <sys/time.h>
#include <cstddef>
#include <cstdlib>
#endif

#ifdef TB_ENERGY
#include <boost/process.hpp>
#endif

#include <iomanip>
#include <sstream>
#include <vector>
#include <iostream>
#include <string>
#include <thread>

namespace TB {

  class Timer{
    private:
      double t0, sum;
      double energyLast_ = 0.0;
#ifdef TB_ENERGY
      double energyAccu_ = 0.0;
      // For accounting for fractional energy readings
      double energyT0_ = 0.0, energyT1_ = 0.0, energyFrac_ = 1.0, energyLeftover_ = 0.0;
      bool   stepFlag_ = false;
#endif
      std::vector<double> laps;
#ifndef MPICODE
      struct timeval TT;
      inline double get() {
        gettimeofday(&TT, (struct timezone *) NULL);
        return (TT.tv_sec)+(TT.tv_usec)*static_cast<double>(0.000001);
      }
#else
      inline double get() {	return MPI_Wtime(); }
#endif

#ifdef TB_ENERGY
      boost::process::ipstream powerScriptInput;
      boost::process::child    powerCollector;
      std::thread updateThread;
      bool isMainRunning;

    public:
      Timer()
        : powerScriptInput(),
          powerCollector("./deltaEnergy.sh", boost::process::std_out > powerScriptInput),
          updateThread([&]() { this->powerDrawLoop(); }),
          isMainRunning(true)
      { };

      ~Timer() {
        powerCollector.terminate();
        isMainRunning = false;
        updateThread.join();
      }

      void powerDrawLoop() {
        double readPower = getPowerDraw();
        while (isMainRunning && powerCollector.running()) {
          this->energyAccu_   += readPower *     energyFrac_ ;
          this->energyLeftover_= readPower *(1.0-energyFrac_);
        }
      }

      inline double getPowerDraw() {
        std::string line;
        double res = -1.0;
        // TODO put this somewhere more sensible
#ifdef MPICODE
        int initialized=0; while(!initialized){ MPI_Initialized(&initialized);}
#endif
        energyT0_ = this->get();
        while (powerScriptInput && std::getline(powerScriptInput, line) && !line.empty()) {
          try {
            res = std::stod(line);
            break;
          } catch (std::invalid_argument e) {
          } catch (std::out_of_range e) {
          }
        };
        energyT1_ = this->get();
        stepFlag_= true;  std::cout<<"getPowerDraw "<<"stepFlag made "<<stepFlag_<<std::endl;
        return res;
      }
#else
    public:
#endif
      inline void   init(){
        sum= 0.0; on();
#ifdef TB_ENERGY
      // Reset all energy accumulation variables
//      energyAccu_ = energyT0_ = energyT1_ = energyFrac_ = energyLeftover_ = 0.0; 
//      stepFlag_ = false; std::cout<<"init() stepFlag made "<< stepFlag_<<std::endl;
#endif
      }
      inline void   on  (){ t0 = get(); }
      // WARNING: tot() and lap() will print, but only init() and on() will reset.
      inline double tot (){ return sum; }
      inline double lap (bool keep=true, bool energy=false, int myRank=0){
        double tt=get(), tr=tt-t0; sum+=tr; if(keep){laps.push_back(tr);};
#ifdef TB_ENERGY
        if (energy){ //&& !myRank) {
          stepFlag_= false;   // Until getPowerDraw() sets it again
          std::cout<<"lap stepFlag made "<< stepFlag_<<std::endl;
          energyFrac_ = abs( (tt - energyT0_)/(energyT1_ - energyT0_) );
          energyFrac_ = std::min( energyFrac_ , 1.0 );
          while(!!stepFlag_ == 0){ }; // Wait for a new power read, which will be fracional
          energyFrac_ = 1.0;
          energyLast_ = energyAccu_;
          this->energyAccu_ = energyLeftover_;
          energyLeftover_ = 0.0;
        }
#endif
        return tr;
      }
      inline double lastEnergyReading() { return energyLast_; }

      // Output timing in a centralized fashion.
      std::string getTimings() {
      	int rank = 0, totalRanks = 1;
      	std::stringstream buf;
#ifdef MPICODE
      	std::vector<double> allResults;
      	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      	MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);
      	double *recvBuf = nullptr;
      	if (rank == 0)
        { allResults.resize(laps.size() * totalRanks);
      	  recvBuf = allResults.data();
      	}
      	MPI_Gather(laps.data(), laps.size(), MPI_DOUBLE, recvBuf, laps.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        std::vector<double> stepMean, stepMax, stepMin;
        std::vector<long>   stepMaxLoc, stepMinLoc;
      	if (rank == 0)
      	{ stepMean.resize(laps.size());
      	  stepMin.resize(laps.size());  stepMinLoc.resize(laps.size());
      	  stepMax.resize(laps.size());  stepMaxLoc.resize(laps.size());

      	  for (size_t l = 0; l < laps.size(); l++)  // Gather some summary statistics
          for (int r = 0; r < totalRanks ; r++)
          { auto curRes = allResults[r * laps.size() + l];
            if (r == 0 || curRes < stepMin[l]) {
          		stepMin[l] = curRes;
          		stepMinLoc[l] = r;
     	      }
	          if (r == 0 || curRes > stepMax[l]) {
        		  stepMax[l] = curRes;
          		stepMaxLoc[l] = r;
	          }
    	      stepMean[l] += curRes / totalRanks;
	        }
      	  // And dump them to as a nicely formatted table.
      	  buf<<"\nMPI Load Imbance\n";
          buf<<std::setw(6)<<"\t" <<"Step"<<"\t" <<std::setw(10)<<"Avg_Time/s"
             <<"\t" <<std::setw(10)<<"Min_Time_%"<<"\t"<<std::setw(10)<<"Min_Rank"
             <<"\t" <<std::setw(10)<<"Max_Time_%"<<"\t"<<std::setw(10)<<"Max_Rank"<<std::endl;
          for (size_t i = 0; i < laps.size(); i++) {
      	    buf<<std::setw(6)<<"\t" <<i<<"\t" <<std::setw(10)<<stepMean[i]
 	             <<"\t"<<std::setw(10)<<100.0*(stepMin[i]/stepMean[i]-1.0)<<"\t"<<std::setw(10)<<stepMinLoc[i]
               <<"\t"<<std::setw(10)<<100.0*(stepMax[i]/stepMean[i]-1.0)<<"\t"<<std::setw(10)<<stepMaxLoc[i]<<std::endl;
	        }
	      }
#else
      	std::vector<double> &allResults = laps;
#endif
      	if (rank == 0) {     	  // Followed by the rest of the data.
      	  buf << "\n{" << std::endl;
      	  for (int r = 0; r < totalRanks; r++) {
      	    buf << "\n\t\"rank" << r << "\": [";
	          for (size_t i = 0; i < laps.size(); i++) {
      	      buf << allResults[r * laps.size() + i] << ((i == laps.size() - 1) ? "" : ", ");
	          }
      	    buf << "]" << ((r == totalRanks-1) ? "" : ", ");
	        }
      	  buf << "\n}" << std::endl;
      	}
      	return buf.str();
      } // END getTimings() function
  }; // END Timer class
} // END TB namespace
#endif

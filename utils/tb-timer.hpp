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
      double t0, sum, tt;
      double energyLast_ = 0.0;
      int globRank = 0, globSize = 1;
#ifdef TB_ENERGY
      int nodeRank = 0, nodeSize = 1;
      int enerInitd = 0;
      double energyAccu_ = 0.0;
      volatile size_t stepFlag_ = 1; // Mark when power read is astride over two laps
      // Account for fractional energy readings across such cases
      double energyT0_ = 0.0, energyT1_ = 0.0, energyFrac_ = 1.0, energyLeftover_ = 0.0;
#ifdef MPICODE
      int initialized = 0;
#endif
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
    private:
      boost::process::ipstream powerScriptInput;
      boost::process::child    powerCollector;
      std::thread updateThread;
      bool isMainRunning;

    public:
      Timer()
        : powerScriptInput(),
          powerCollector("./deltaEnergy.sh", boost::process::std_out > powerScriptInput),
//          updateThread([&]() { this->powerDrawLoop(); }),
          isMainRunning(true)
      { enerInit(); };
//      { std::cout<<"constructor"<<std::endl;};

      ~Timer() {
        powerCollector.terminate();
        isMainRunning = false;
        updateThread.join();
      }

      void enerInit(){
       enerInitd=1;
#ifdef MPICODE
        while(!initialized){ MPI_Initialized(&initialized); }// std::cout<<(initialized?"i":"n");} std::cout<<std::endl;
      	MPI_Comm_rank(MPI_COMM_WORLD, &globRank);
      	MPI_Comm_size(MPI_COMM_WORLD, &globSize);
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
        MPI_Comm_rank(local_comm, &nodeRank);
        MPI_Comm_size(local_comm, &nodeSize);
#endif
        if(!nodeRank) this->updateThread = std::thread([&]() { this->powerDrawLoop(); });
      }

      void powerDrawLoop() {
        while (isMainRunning && powerCollector.running()) {
          energyFrac_ = 1.0;
          energyT0_ = this->get();
          double readPower = getPowerDraw();
          if( (!stepFlag_) && (tt >= energyT0_) ){
            energyT1_   = this->get();
            energyFrac_ = ( tt-energyT0_ )/( energyT1_-energyT0_ );
            // TODO Debug print, remove
            std::cout <<globRank<<" "<<globSize<<" " <<nodeRank<<" "<<nodeSize<<"T0, Dtt, DT1, EF: "<<energyT0_<<", "<<tt-energyT0_<<", "<<energyT1_-energyT0_<<","<<energyFrac_<< std::endl;
            stepFlag_  = 1;
          }
          this->energyAccu_   += readPower *     energyFrac_ ;
          this->energyLeftover_= readPower *(1.0-energyFrac_);
          std::cout <<"Powers at this lap: "<< readPower<<" "<<this->energyAccu_<<", "<<this->energyLeftover_<< std::endl;
          energyFrac_= 1.0;
        }
      }

      inline double getPowerDraw() {
        std::string line;
        double res = -1.0;
        while (powerScriptInput && std::getline(powerScriptInput, line) && !line.empty()) {
          try {
            res = std::stod(line);
            break;
          } catch (std::invalid_argument e) {
          } catch (std::out_of_range e) {
          }
        };
        return res;
      }
#else
    public:
#ifdef MPICODE
      Timer(){
        while(!initialized){ MPI_Initialized(&initialized); }// std::cout<<(initialized?"i":"n");} std::cout<<std::endl;
      	MPI_Comm_rank(MPI_COMM_WORLD, &globRank);
      	MPI_Comm_size(MPI_COMM_WORLD, &globSize);
      };
#endif

#endif
      inline void   init(){
        sum= 0.0; on();
#ifdef TB_ENERGY        // Reset all energy accumulation variables
//        if( !enerInitd) enerInit();
        energyAccu_ = energyT0_ = energyT1_ = 0.0;
        energyFrac_ = 1.0; stepFlag_ = 0;
        energyLeftover_ = 0.0; // An init must zero also this... but is the algorithm correct like this?
#endif
      }
      inline void   on  (){ t0 = get(); }
      // WARNING: tot() and lap() will print, but only init() and on() will reset.
      inline double tot (){ return sum; }
      inline double lap (bool keep=true, bool energy=false, int myRank=0){
        tt = get();
        double tr=tt-t0; sum+=tr; if(keep){laps.push_back(tr);};
#ifdef TB_ENERGY
        if (energy && !nodeRank) {
          stepFlag_= 0;   // Until getPowerDraw() sets it again
          // Wait for a power read. You alredy saved the time.
          //  TODO Init after each step seems more correct,
          //  not sure what happens to the extra time otherwise.
          while(1) { if (stepFlag_) break; };  // TODO W/O cout ... it loops forver. _shrug_ 
//          std::cout<<std::endl; // DEBUG PRINT
          this->energyLast_ = this->energyAccu_;
          this->energyAccu_ = this->energyLeftover_;
          this->energyLeftover_ = 0.0;
        }
#endif
        return tr;
      }
      inline double lastEnergyReading() { return energyLast_; }

      // Output timing in a centralized fashion.
      std::string getTimings() {
      	std::stringstream buf;
#ifdef MPICODE
      	std::vector<double> allResults;
      	double *recvBuf = nullptr;
      	if (globRank == 0)
        { allResults.resize(laps.size() * globSize);
      	  recvBuf = allResults.data();
      	}
      	MPI_Gather(laps.data(), laps.size(), MPI_DOUBLE, recvBuf, laps.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        std::vector<double> stepMean, stepMax, stepMin;
        std::vector<long>   stepMaxLoc, stepMinLoc;
      	if (globRank == 0)
      	{ stepMean.resize(laps.size());
      	  stepMin.resize(laps.size());  stepMinLoc.resize(laps.size());
      	  stepMax.resize(laps.size());  stepMaxLoc.resize(laps.size());

      	  for (size_t l = 0; l < laps.size(); l++)  // Gather some summary statistics
          for (int r = 0; r < globSize ; r++)
          { auto curRes = allResults[r * laps.size() + l];
            if (r == 0 || curRes < stepMin[l]) {
          		stepMin[l] = curRes;
          		stepMinLoc[l] = r;
     	      }
	          if (r == 0 || curRes > stepMax[l]) {
        		  stepMax[l] = curRes;
          		stepMaxLoc[l] = r;
	          }
    	      stepMean[l] += curRes / globSize;
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
      	return buf.str();
      } // END getTimings() function
  }; // END Timer class
} // END TB namespace
#endif

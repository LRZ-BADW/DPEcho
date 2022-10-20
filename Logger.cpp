//   Copyright(C) 2020 Fabio Baruffa, Intel Corp.
//   Copyright(C) 2021 Salvatore Cielo, LRZ
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
//  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
//  language governing permissions and limitations under the License.

#include "Logger.hpp"
#include "echo.hpp"
#include <cstdarg>
#include <cstdio>

const string mytag(const string val){
  auto start = val.find(" ")+1, end = val.find("(");  // for funcs w type
  start = (start > end) ? 0 : start;                  // fix for funcs w/o type
  return "["+val.substr(start, end - start)+"] ";
}

bool Logger::instanceFlag = false;
Logger* Logger::single = NULL;

Logger* Logger::getInstance ( int *argc, char*** argv) {
  if(!instanceFlag){
    single = new Logger(argc, argv);
    instanceFlag=true;
    return single;
  } else {
    return single;
  }
}

Logger* Logger::getInstance () {
  if(!instanceFlag) {
    cerr << "ERROR" << ": Logger::getInstance(...,...) is not initialized: call getInstance(argc,argv) first!"  << std::endl;
    abort();
  }
  return single;
}

Logger::Logger(int *argc, char*** argv) {
  verb_ = cverb_ = 0;
  pario_ = false;
#ifdef MPICODE
  MPI_Init(argc, argv);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs_);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank_);
  if(master()){ffs_.open("log", ios_base::out);}
  Logo();
  (*this)<<TAG<< "\n\tVersion: MPI with "<< nprocs_ << " tasks, ";
#else
  argc = argc; argv = argv;
  nprocs_=1; myrank_=0;
  ffs_.open("log", ios_base::out);
  Logo();
  *(this)<<TAG <<"\n\tVersion  : Serial";
#endif
#ifndef NDEBUG
  *(this)<<"\n\tDebug    : ON";
#endif
  *(this)<<"\n\tFields   : "<<FLD_TOT;
  *(this)<<"\n\tPhysics  : "<< (PHYSICS ? "GRMHD":"MHD");
#ifdef SINGLE_PRECISION
  *(this)<<"\n\tPrecision: Single";
#else
  *(this)<<"\n\tPrecision: Double";
#endif
  *(this)<<"\n\tRK Order : "<<NRK;
  *(this)<<"\n\tRec Order: "<<REC_ORDER;
#if REC_ORDER==5
  *(this)<<"\n\tRec Type : MP5";
#else
  *(this)<<"\n\tRec Type : "<<RECONSTR;
#endif
  *(this)<<"\n\tNGC      : "<<NGC;
  this->fl();
  barrier();
  time.init();
}

Logger::~Logger(){  exit(EXIT_SUCCESS); }

bool Logger::master() {
  if(myrank_ == 0) return true; else return false;
}

void Logger::Info(int level, const char *msg, ... ) {
    va_list argptr;
    char message[2048];
    if(level > verb_) return;

    va_start(argptr,msg);
    vsprintf(message, msg, argptr );
    va_end(argptr);

    if( master() ) {
      cout << "I: ";
      for( int i = 0; i < level; i++ ) cout << " ";
      cout << message;
       message[0] = '\0';
      cout << flush;
    }
    barrier();
}

void Logger::Write(const char *msg, ... ){
    va_list argptr;
    char message[2048];

    va_start(argptr, msg);
    vsprintf(message, msg, argptr );
    va_end(argptr);

    cout << message;
    cout << std::flush;
    barrier();
}

void Logger::Error (const char *msg, ... ) {
    va_list argptr;
    char message[2048];
    va_start ( argptr, msg);
    vsprintf ( message, msg, argptr );
    va_end(argptr);

    if( master() ) {
      cerr << "ERROR" << ": FatalException: " << message << std::endl;
    }
    message[0] = '\0';

#ifdef MPICODE
    int rv = -1;
    MPI_Abort(MPI_COMM_WORLD,rv);
    MPI_Finalize();
#endif
    abort();
}

void Logger::Debug(const char *msg, ... ) {
#ifndef NDEBUG
  va_list argptr;
  char message[2048];
  va_start(argptr,msg);
  vsprintf(message, msg, argptr );
  va_end(argptr);
  cout << "D: ";
  cout << message;
#endif
}

void Logger::barrier(){
#ifdef MPICODE
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

void Logger::cleanup(){
  barrier();
  *this<<TAG<<"Final cleanup. Total runtime [s]: "<< time.lap(); fl();
#ifdef MPICODE
  MPI_Finalize();
#endif
  //el();
  if(master()){ ffs_.close();}
  delete this;
}

void Logger::Logo(){
  *this+0<<"    ____   ____   ______       __          \n";
  *this+0<<"   / __ \\ / __ \\ / ____/_____ / /_   ____  \n";
  *this+0<<"  / / / // /_/ // __/  / ___// __ \\ / __ \\ \n";
  *this+0<<" / /_/ // ____// /___ / /__ / / / // /_/ / \n";
  *this+0<<"/_____//_/    /_____/ \\___//_/ /_/ \\____/  \n";
  fl();
}

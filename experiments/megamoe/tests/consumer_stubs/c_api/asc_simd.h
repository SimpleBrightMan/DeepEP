#pragma once
constexpr unsigned PIPE_FIX=0,PIPE_MTE2=1,PIPE_MTE3=2;
namespace CreditModel { void Enqueue(unsigned pipe,unsigned flag,bool arrive); }
inline void asc_sync_intra_arrive(unsigned pipe,unsigned flag) { CreditModel::Enqueue(pipe,flag,true); }
inline void asc_sync_intra_wait(unsigned pipe,unsigned flag) { CreditModel::Enqueue(pipe,flag,false); }

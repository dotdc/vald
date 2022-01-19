#include <stdio.h>
#include <stdlib.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include "Capi.h"

FaissStruct* faiss_create_index() {
  int d = 64;      // dimension
  int nlist = 100;
  int m = 8;
  faiss::IndexFlatL2 *quantizer = new faiss::IndexFlatL2(d);
  faiss::IndexIVFPQ *index = new faiss::IndexIVFPQ(quantizer, d, nlist, m, 8);
  FaissStruct *st = new FaissStruct{
    static_cast<FaissQuantizer>(quantizer),
    static_cast<FaissIndex>(index)
  };

  printf(__FUNCTION__);
  fflush(stdout);
  return st;
}

void faiss_train(const FaissStruct* st) {
  int d = 64;      // dimension
  int nb = 10000;  // database size
  int nq = 1000;   // nb of queries
  float *xb = new float[d * nb];
  float *xq = new float[d * nq];
  for(int i = 0; i < nb; i++) {
      for(int j = 0; j < d; j++) xb[d * i + j] = drand48();
      xb[d * i] += i / 1000.;
  }
  for(int i = 0; i < nq; i++) {
      for(int j = 0; j < d; j++) xq[d * i + j] = drand48();
      xq[d * i] += i / 1000.;
  }

  (static_cast<faiss::IndexIVFPQ*>(st->faiss_index))->train(nb, xb);

  printf(__FUNCTION__);
  fflush(stdout);
  return;
}

void faiss_free(FaissStruct* st) {
  free(st);

  printf(__FUNCTION__);
  fflush(stdout);
  return;
}

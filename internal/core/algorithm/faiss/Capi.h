#ifdef __cplusplus
extern "C" {
#endif
  #include <stdio.h>
  #include <stdlib.h>

  typedef void* FaissQuantizer;
  typedef void* FaissIndex;
  typedef struct {
    FaissQuantizer  faiss_quantizer;
    FaissIndex      faiss_index;
  } FaissStruct;

  FaissStruct* faiss_create_index();
  void faiss_train(const FaissStruct* st);
  void faiss_free(FaissStruct* st);
#ifdef __cplusplus
}
#endif

from rag.sparse_index import SparseIndex
from rag.dense_index import DenseIndex 
from rag.fusion import reciprocal_rank_fusion

# combines all retrieval methods into a single object with a consistent interface
class HybridRetriever:
    def __init__(self, chunks):
        self.chunks = chunks 
        self.sparse_index = SparseIndex(chunks)
        self.dense_index = DenseIndex(chunks)

    # individual thin wrappers for the standalone search methods 
    # this allows us to perform individual component searches on the chunks 
    def search_sparse(self, query, k=5):
        return self.sparse_index.search(query, k)
        
    def search_dense(self, query, k=5):
        return self.dense_index.search(query, k)

    # one giant function to call the individual search methods and create a fused ranking
    def search_hybrid(self, query, k=5, overfetch=10):
        # the idea is we want to overfetch a certain number chunks and then take the top k from them
        sparse_result = self.search_sparse(query, k=overfetch)
        dense_result = self.search_dense(query, k=overfetch)
        result = reciprocal_rank_fusion([sparse_result, dense_result], k=60)
        return result[:k]

    def get_texts(self, results):
        # pull ot the chunk texts from the results - used for the next part which is the pipeline
        return [chunk["text"] for chunk, _score in results]
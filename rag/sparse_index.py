from rank_bm25 import BM25Okapi

class SparseIndex:
    def __init__(self, chunks):
        """
        chunks: list of dicts from chunk_documents(), each with a "text" key
        """
        # we want to use these chunks in different methods so intialise it 
        self.chunks = chunks 

        # BM25 computed on individual words so we want to tokenise strings into a list of words
        tokenised_chunks = [chunk['text'].lower().split() for chunk in chunks]

        # intialise the object to calculate BM25 
        self.bm25 = BM25Okapi(tokenised_chunks) 

    def search(self, query, k=5):
        """
        Returns the top-k chunks (full dicts) ranked by BM25 score, along with their scores.
        """
        tokenised_query = query.lower().split()
        scores = self.bm25.get_scores(tokenised_query) # 1 score per chunk
        paired = zip(self.chunks, scores) 

        # pair each chunk with its score, sort descending, take top-k
        ranked = sorted(paired, key=lambda pair: pair[1], reverse=True) 

        return ranked[:k] # list of (chunk_dict, score) tuples
from sentence_transformers import SentenceTransformer 
import faiss 

class DenseIndex:
    def __init__(self, chunks, model_name="all-MiniLM-L6-v2"):
        # intialise the chunks that we will use throughout 
        # extract the text for each chunk to create a list of chunk strings - this is what we will pass to be embedded
        self.chunks = chunks 
        text_chunks = [chunk['text'] for chunk in chunks]

        # intialise the embedding model we will be using 
        self.model = SentenceTransformer(model_name)

        # embed the text chunks into vectors - a vector for each chunk
        embedded_chunks = self.model.encode(text_chunks)

        # takes the embedded chunks and normalises them to a magnitude of 1
        faiss.normalize_L2(embedded_chunks) # normalises in place
        
        # create a data structure to store all the chunk vectors inside 
        self.index = faiss.IndexFlatIP(embedded_chunks.shape[1]) 

        # populate the index with the normalised vectors 
        self.index.add(embedded_chunks)

    def search(self, query, k=5):
        # embed the query using the same embedding model as the chunks
        embedded_query = self.model.encode([query])

        # normalise the query 
        faiss.normalize_L2(embedded_query)

        # now perform a search on the index of chunks against the query by computing cosine similarity
        scores, indices = self.index.search(embedded_query, k) 
        # this is exactly why we store the embeded chunks in an index - so we can search it with respect to a query

        # to store the (chunk_dict, score) tuples
        result = []

        # populate the result list
        for score, chunk_idx in zip(scores[0], indices[0]):
            result.append((self.chunks[chunk_idx], float(score)))
        return result
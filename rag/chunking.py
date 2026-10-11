import tiktoken
enc = tiktoken.get_encoding('gpt2') # we want to encode to tokens so we can guage how large each chunk should be to fit the 1024 context window limit
import os

def chunk_text(text, chunk_size=150, overlap=30):
    token_ids = enc.encode(text) # encodes the text into a list of integer tokens
                                 # e.g. "the cat sat on the math" becomes something like [1169, 3797, 3332, 319, 262, 2603]
                                 # this allows us to chunk by tokens to guarantee we never exceed the context window 
    chunks = [] # store the 150 token long chunks
    start = 0 # starting at the beginning 

    # converts text into tokens and splits this list of tokens into chunks of 150 tokens
    while start < len(token_ids):
        end = start + chunk_size # gets the end of the first chunk 
        chunk_ids = token_ids[start:end] # indexes into the token_ids to get the chunk of tokens
        chunk_text_str = enc.decode(chunk_ids) # decode the chunk into a string
        chunks.append(chunk_text_str) # append that string to chunks so we have a list of chunk strings
        if end >= len(token_ids): # break out once reached the end of exceeded the end
            break
        start = end - overlap # step forward, but re-include the overlap region
    return chunks


# chunk multiple documents making use of chunk_text - here documents are stored as a dict with its id and text content
# wraps each of the chunks with additional metadata such as the document it comes from and its own uique id 
def chunk_documents(documents):
    all_chunks = []
    for doc in documents: # iterate over all the dictionaries in documents - stores 
        pieces = chunk_text(doc["text"]) # obtain the text for that document and chunk it into pieces
        # iterate over every chunk of the document
        # append to a list of dictionaries - each dictionary stores the chunk_id, doc_id and the text chunk
        # this generates a giant list all chunks along with where it came from (doc_id) and it's unique identifier (doc_id)
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "chunk_id": f"{doc['id']}_chunk{i}",
                "doc_id": doc["id"],
                "text": piece,
            })
    return all_chunks

# document loader that takes a document and converts it into a dictionary consisting of its id and contents
def load_documents(raw_dir="data/raw"):
    documents = [] # list of dicts for each doc
    for filename in sorted(os.listdir(raw_dir)): # iterate over a list of all filenames in the directory - os.listdir() gives a list of files in the given directory
        if not filename.endswith(".txt"): # if not a txt file then ignore
            continue
        path = os.path.join(raw_dir, filename) # create the path to access the file
        with open(path, "r", encoding="utf-8") as f: # open the txt file
            text = f.read() # read the txt file 
        documents.append({"id": filename, "text": text}) # store that file's contents as a dictionary and append to the list of txt file dicts
    return documents
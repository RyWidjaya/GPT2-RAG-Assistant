def answer_question(retriever, model, enc, device, eos_id, max_seq_len, question, k=3, method="hybrid", max_context_tokens=400):

    from generate import ask

    # search our chunks depending on the method specified 
    if method == "sparse":
        results = retriever.search_sparse(question, k=k)
    elif method == "dense":
        results = retriever.search_dense(question, k=k)
    elif method == "hybrid":
        results = retriever.search_hybrid(question, k=k)
    
    # the search will return a list of tuples of the top chunks - iterate over this list of tuples and extract the text from the chunks 
    # we will then return a string of chunk texts separated by a new line for each chunk
    chunk_texts = [chunk["text"] for chunk, _score in results]
    context = "\n".join(chunk_texts)

    # defensive: even with sensible chunk_size, joining several chunks could occasionally push context too long. 
    # Truncate defensively by token count before it ever reaches ask().
    # We will encode our text into tokens to see if its sie exceeds the token size context window limit 
    # we then truncate as needed and convert back into text after truncating the text so its tokenised vsriant fits the context window size
    context_ids = enc.encode(context) 
    if len(context_ids) > max_context_tokens: 
        context_ids = context_ids[:max_context_tokens] # truncate defensively just in case it exceeds the content limit
        context = enc.decode(context_ids)
    
    # query our transformer model using the context and question
    # remember our prompt being fed into the model is structured as f"Context: {context}\nQuestion: {question}\nAnswer:"
    answer = ask(model, enc, device, eos_id, max_seq_len, context, question)
    
    # return the answer, the context and the chunks for evaluation 
    return {
        "answer": answer,
        "context_used": context,
        "retrieved_chunks": results,
        }
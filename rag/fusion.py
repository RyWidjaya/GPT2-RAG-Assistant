def reciprocal_rank_fusion(retrieval_results, k):

    rrf_scores = {}
    chunks = {}

    # iterate over every score list [(dict, score), ...]
    for result in retrieval_results:
        # iterate into the list - iterate over the list of dicts, access the chunk id and score then compute the RRF contribution
        for rank, chunk in enumerate(result, start=1):
            rrf_scores[chunk[0]['chunk_id']] = rrf_scores.get(chunk[0]['chunk_id'], 0) + 1 / (k + rank)
            chunks[chunk[0]['chunk_id']] = chunk[0] # add chunk_dict to a dict so we can access by chunk_id

    fusion = [(chunks[key], rrf_scores[key]) for key in rrf_scores]
    return sorted(fusion, key=lambda x: -x[1])
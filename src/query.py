
import asyncio
import time
from llama_index.core import VectorStoreIndex

from src.config import DEFAULT_TOP_K
import json


def extract_evidence(nodes):

    evidence_list = []

    for index, node_with_score in enumerate(nodes, start=1):

        node = node_with_score.node

        metadata = node.metadata or {}

        evidence = {

            "evidence_id": f"E{index}",

            "chunk_id": node.node_id,

            "filename": metadata.get("file_name"),

            "file_path": metadata.get("file_path"),

            "section": metadata.get("header_path"),

            "start_char_idx": node.start_char_idx,

            "end_char_idx": node.end_char_idx,

            "exact_text": node.get_content(
                metadata_mode="none"
            ),

            "score": node_with_score.score

        }

        evidence_list.append(evidence)

    return evidence_list




async def retiver_query(
    query: str,
    index: VectorStoreIndex,
    llm,
    on_token=None,
    on_evidence=None,
):

    total_start = time.perf_counter()

    try:

        # --------------------------------
        # 1. Initialize Qdrant and Index
        # --------------------------------

        start = time.perf_counter()

        index_init_time = time.perf_counter() - start

        # --------------------------------
        # 2. Initialize Retriever
        # --------------------------------

        start = time.perf_counter()

        dense_retriever = index.as_retriever(
            similarity_top_k=DEFAULT_TOP_K
        )

        retriever_init_time = time.perf_counter() - start

        # --------------------------------
        # 3. Retrieve Documents
        # --------------------------------

        start = time.perf_counter()

        nodes = await asyncio.to_thread(
            dense_retriever.retrieve,
            query
        )

        retrieval_time = time.perf_counter() - start

        # --------------------------------
        # 4. Extract Evidence
        # --------------------------------

        start = time.perf_counter()

        query_input = extract_evidence(nodes)

        evidence_time = time.perf_counter() - start

        # --------------------------------
        # 5. Print / stream Retrieved Evidence
        # --------------------------------

        print("\n========== RETRIEVED EVIDENCE ==========")

        print(json.dumps(
            query_input,
            indent=4,
            default=str
        ))

        if on_evidence is not None:
            await on_evidence(query_input)

        # --------------------------------
        # 6. LLM Generation
        # --------------------------------

        llm_start = time.perf_counter()
        response = await llm.generate_decision(
            query=query,
            evidence=query_input,
            on_token=on_token
        )

        llm_time = time.perf_counter() - llm_start

        # --------------------------------
        # 7. Total Execution Time
        # --------------------------------

        total_time = time.perf_counter() - total_start

        # --------------------------------
        # 8. Performance Report
        # --------------------------------

        print("\n")
        print("=" * 55)
        print("             RAG PERFORMANCE REPORT")
        print("=" * 55)

        print(f"Index initialization : {index_init_time:.4f} sec")

        print(f"Retriever creation   : {retriever_init_time:.4f} sec")

        print(f"Document retrieval   : {retrieval_time:.4f} sec")

        print(f"Evidence extraction  : {evidence_time:.4f} sec")

        print(f"LLM generation       : {llm_time:.4f} sec")

        print("-" * 55)

        print(f"Total execution      : {total_time:.4f} sec")

        print(f"Retrieved chunks     : {len(nodes)}")

        print("=" * 55)

        print("\nLLM RESPONSE:")
        print(response)

        return {
            "decision": response,
            "evidence": query_input,
            "timings": {
                "retrieval_sec": retrieval_time,
                "generation_sec": llm_time,
                "total_sec": total_time,
            },
        }
    except Exception as e:
        raise e




# if __name__ == "__main__":

#     asyncio.run(
#         retiver_query(
#             "Can I return unused, unopened headphones delivered 20 days ago? I have the original box and receipt."
#         )
#     )

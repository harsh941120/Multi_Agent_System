import time
from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain
 
 
def run_research_pipeline(topic: str) -> dict:
 
    state = {}
    pipeline_start = time.time()
 
    # search agent working
    print("\n" + "=" * 50)
    print("step 1 - search agent is working ....")
    print("=" * 50)
 
    t0 = time.time()
    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [('user', f"Find recent , reliable and detailed information about {topic}")]
    })
    state["search_results"] = search_result['messages'][-1].content
    print(f"\n[TIMING] Search agent took {time.time() - t0:.2f}s")
 
    print("\n search result ", state['search_results'])
 
    # step 2 -reader agent
    print("\n" + "=" * 50)
    print("step 2 - reader  agent is scrapping top resources ....")
    print("=" * 50)
 
    t0 = time.time()
    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
    })
    state['scraped_content'] = reader_result['messages'][-1].content
    print(f"\n[TIMING] Reader agent took {time.time() - t0:.2f}s")
 
    print("\nscraped content: \n", state['scraped_content'])
 
    # step 3 -writer chain
    print("\n" + "=" * 50)
    print("step 3 - writer is drafting the report ....")
    print("=" * 50)
 
    research_combined = (
        f"SEARCH RESULT : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']} \n\n"
    )
 
    t0 = time.time()
    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })
    print(f"\n[TIMING] Writer chain took {time.time() - t0:.2f}s")
 
    print("\n Final Report\n", state['report'])
 
    # critic report
    print("\n" + "=" * 50)
    print("step 4 - critic is reviwing  the report ....")
    print("=" * 50)
 
    t0 = time.time()
    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })
    print(f"\n[TIMING] Critic chain took {time.time() - t0:.2f}s")
 
    print("\n critic report \n", state["feedback"])
 
    print(f"\n[TIMING] TOTAL pipeline time: {time.time() - pipeline_start:.2f}s")
 
    return state
 
 
if __name__ == "__main__":
    topic = input("\n Enter a research topic :")
    run_research_pipeline(topic)
 
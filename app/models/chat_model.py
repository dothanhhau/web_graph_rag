from app.__init__ import chain

def get_chat_response(query):
    try:
        result = chain.invoke({"query": query})
        cypher = result.get("intermediate_steps", [{}])[0].get("query", "")
        answer = result.get("result", "")
    except Exception as e:
        cypher = ''
        answer = 'Câu truy vấn lỗi cú pháp'
    return cypher, answer

def generate_cypher(query):
    args = {"question": query, "schema": chain.graph_schema}  # Cung cấp schema của đồ thị
    generated_cypher = chain.cypher_generation_chain.invoke(args)
    return generated_cypher

def query_neo4j(cypher):
    if chain.cypher_query_corrector:
        cypher = chain.cypher_query_corrector(cypher)
    context = chain.graph.query(cypher)[: chain.top_k]
    return context

def generate_answer(query, res_cypher):
    qa_input = {"question": query, "context": res_cypher}
    final_result = chain.qa_chain.invoke(qa_input)
    return final_result
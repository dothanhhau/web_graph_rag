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
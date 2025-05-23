import os
from flask import Flask
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain.prompts.prompt import PromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
load_dotenv()

graph = Neo4jGraph(
    url=str(os.getenv('URI_NEO4J')), 
    username=str(os.getenv('USER_NEO4J')), 
    password=str(os.getenv('PASSWORD_NEO4J')), 
    database=str(os.getenv('DB_NEO4J')), 
    enhanced_schema=True
)

# Prompt cho tác vụ tìm kiếm(mới)
CYPHER_GENERATION_TEMPLATE = """
Task: Sinh truy vấn Cypher để truy xuất dữ liệu từ đồ thị tri thức.

Instructions:
- Phân tích kỹ câu hỏi và xác định các thành phần quan trọng như thực thể (node), quan hệ (edge), và thuộc tính (property).
- Chỉ sử dụng các loại node, quan hệ và thuộc tính có trong schema bên dưới. Không thêm bất kỳ phần tử nào không có trong schema.
- Các nhãn node được đặt bằng tiếng Việt (ví dụ: :Chương_trình, :Chủ_thể, :Học, :Quy_định...).
- Các loại quan hệ cũng được đặt bằng tiếng Việt (ví dụ: :ÁP_DỤNG, :HỌC, :QUY_ĐỊNH...).
- Mọi truy vấn **phải trả về đầy đủ tất cả thuộc tính** của node và relationship bằng `properties(...)`. Nếu cần, sử dụng `type(...)` để lấy tên quan hệ.
- **Chỉ được sử dụng phép so sánh `CONTAINS` trong mệnh đề WHERE. Không được sử dụng phép gán bằng (`=`).**
- Khi cần truy vấn nhiều mẫu (nhiều thực thể và quan hệ khác nhau), hãy dùng nhiều câu `OPTIONAL MATCH` tách riêng, sau đó gom kết quả lại bằng `collect(...)` và chỉ viết một câu `RETURN` duy nhất ở cuối.
- Không viết nhiều câu `OPTIONAL MATCH ... RETURN` liên tiếp. Luôn gom kết quả lại để trả về một lần duy nhất.
- Không viết nhiều truy vấn MATCH...RETURN liên tiếp mà không tách bằng dòng trắng hoặc dùng `UNION`.
- Không sử dụng `UNION` nếu có thể gộp các mẫu bằng `OPTIONAL MATCH`.
- Không viết lời giải thích, ghi chú hoặc định nghĩa. Chỉ trả về truy vấn Cypher.

Schema đồ thị:
{schema}

Ví dụ truy vấn:

# Các phương thức đóng học phí bao gồm các phương thức nào?
OPTIONAL MATCH (n)-[r]->(m)
WHERE n.id CONTAINS 'Phương Thức Đóng Học Phí'
RETURN 
  properties(n) AS n_properties, 
  type(r) AS r_type, 
  properties(r) AS r_properties, 
  properties(m) AS m_properties

# Học phần bắt buộc là gì?
OPTIONAL MATCH (n)
WHERE n.id CONTAINS 'Học Phần Bắt Buộc'
RETURN properties(n) AS n_properties

# Khi nào sinh viên bị buộc thôi học?
OPTIONAL MATCH (n)-[r1:`BỊ_BUỘC_THÔI_HỌC`]->(m)
WHERE n.id CONTAINS 'Sinh Viên'
RETURN 
  properties(n) AS n_properties, 
  type(r1) AS r1_type, 
  properties(r1) AS r1_properties,
  properties(m) AS m_properties

# Khối lượng học tập của các ngành là bao nhiêu?
OPTIONAL MATCH (n)
WHERE n.id CONTAINS 'Khối Lượng Học Tập'

OPTIONAL MATCH (n2:Ngành_đào_tạo)-[r]->(m)
WHERE m.id CONTAINS 'Khối Lượng Học Tập'

RETURN 
  collect(DISTINCT properties(n)) AS khái_niệm,
  collect(DISTINCT properties(n2)) AS ngành_đào_tạo,
  collect(DISTINCT type(r)) AS loại_quan_hệ,
  collect(DISTINCT properties(r)) AS thuộc_tính_quan_hệ,
  collect(DISTINCT properties(m)) AS khái_niệm_liên_quan

The question is:
{question}
"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
	input_variables=["schema", "question"],
	template=CYPHER_GENERATION_TEMPLATE
)

# Mô hình deepseek
os.environ["DEEPSEEK_API_KEY"] = str(os.getenv("DEEPSEEK_API_KEY"))
llm = ChatDeepSeek(
	model="deepseek-chat",# deepseek-reasoner, deepseek-chat
	# temperature=1,
	# max_tokens=8000,
	# timeout=None,
	# max_retries=2,
)

chain = GraphCypherQAChain.from_llm(
	llm,
	graph=graph,
	verbose=True,
	top_k=5,
	allow_dangerous_requests=True,
	validate_cypher=True,
	return_intermediate_steps=True,
	cypher_prompt=CYPHER_GENERATION_PROMPT,
)

def create_app():
    app = Flask(__name__, template_folder='views')

    from app.controllers.chat_controller import chat_bp
    app.register_blueprint(chat_bp)

    return app
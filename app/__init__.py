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

CYPHER_GENERATION_TEMPLATE = """
Task: Sinh truy vấn Cypher để truy xuất dữ liệu từ đồ thị tri thức.

Instructions:
- Phân tích kỹ câu hỏi và xác định các thành phần quan trọng như thực thể (node), quan hệ (edge), và thuộc tính (property).
- Chỉ sử dụng các loại node, quan hệ và thuộc tính có trong schema bên dưới. Không thêm bất kỳ phần tử nào không có trong schema.
- Không sử dụng nhãn node trong MATCH (chỉ viết OPTIONAL MATCH (n), không OPTIONAL MATCH (n:Label)).
- Các loại quan hệ được đặt bằng tiếng Việt (ví dụ: :ÁP_DỤNG, :HỌC, :QUY_ĐỊNH...).
- Tất cả truy vấn đều phải dùng `OPTIONAL MATCH`. Không sử dụng `MATCH`.

- Mọi truy vấn **phải trả về đầy đủ tất cả thuộc tính** của node và relationship bằng `properties(...)`. Nếu cần, sử dụng `type(...)` để lấy tên quan hệ.
- **Chỉ được sử dụng phép so sánh `CONTAINS` trong mệnh đề WHERE, và luôn sử dụng `toLower(...)` ở cả hai vế.**
  Ví dụ: `toLower(n.id) CONTAINS toLower('từ khóa')`

- Nếu câu hỏi chỉ yêu cầu thông tin về một khái niệm hoặc thực thể đơn lẻ → sử dụng mẫu:
  `OPTIONAL MATCH (n) WHERE ...`
- Nếu câu hỏi yêu cầu mối liên hệ hoặc dữ kiện liên quan giữa các khái niệm → sử dụng mẫu:
  `OPTIONAL MATCH (n)-[r]-(m) WHERE ...`

- Khi cần truy vấn nhiều mẫu (nhiều thực thể và quan hệ khác nhau), hãy dùng nhiều câu `OPTIONAL MATCH` tách riêng, sau đó gom kết quả lại bằng `collect(...)` và chỉ viết một câu `RETURN` duy nhất ở cuối.
- Không sử dụng `UNION` trong bất kỳ trường hợp nào.
- Không viết nhiều câu `OPTIONAL MATCH...RETURN` liên tiếp. Luôn gom kết quả lại để trả về một lần duy nhất bằng một câu `RETURN`.
- Không viết lời giải thích, ghi chú hoặc định nghĩa. Chỉ trả về truy vấn Cypher.

- Nếu không xác định được thực thể, quan hệ hoặc thuộc tính nào trong schema liên quan đến câu hỏi → KHÔNG sinh truy vấn.
- Trong trường hợp đó, chỉ trả về: `RETURN []`

Schema đồ thị:
{schema}

Ví dụ truy vấn:

# Các phương thức đóng học phí bao gồm các phương thức nào?
OPTIONAL MATCH (n)-[r]->(m)
WHERE toLower(n.id) CONTAINS toLower('Phương Thức Đóng Học Phí')

RETURN 
  collect(DISTINCT properties(n)) AS n_properties, 
  collect(DISTINCT type(r)) AS r_type, 
  collect(DISTINCT properties(r)) AS r_properties, 
  collect(DISTINCT properties(m)) AS m_properties

# Học phần bắt buộc là gì?
OPTIONAL MATCH (n)
WHERE toLower(n.id) CONTAINS toLower('Học Phần Bắt Buộc')

RETURN collect(DISTINCT properties(n)) AS n_properties

# Khi nào sinh viên bị buộc thôi học?
OPTIONAL MATCH (n)-[r1]->(m)
WHERE toLower(n.id) CONTAINS toLower('Sinh Viên') AND type(r1) = 'BỊ_BUỘC_THÔI_HỌC'

RETURN 
  collect(DISTINCT properties(n)) AS n_properties, 
  collect(DISTINCT type(r1)) AS r1_type, 
  collect(DISTINCT properties(r1)) AS r1_properties,
  collect(DISTINCT properties(m)) AS m_properties

# Khối lượng học tập của các ngành là bao nhiêu?
OPTIONAL MATCH (n)
WHERE toLower(n.id) CONTAINS toLower('Khối Lượng Học Tập')

OPTIONAL MATCH (n2)-[r]->(m)
WHERE toLower(m.id) CONTAINS toLower('Khối Lượng Học Tập')

RETURN 
  collect(DISTINCT properties(n)) AS khái_niệm,
  collect(DISTINCT properties(n2)) AS ngành_đào_tạo,
  collect(DISTINCT type(r)) AS loại_quan_hệ,
  collect(DISTINCT properties(r)) AS thuộc_tính_quan_hệ,
  collect(DISTINCT properties(m)) AS khái_niệm_liên_quan

# Những quy định liên quan đến điều kiện tốt nghiệp là gì?
OPTIONAL MATCH (n)-[r]-(m)
WHERE toLower(n.id) CONTAINS toLower('Điều Kiện Tốt Nghiệp')

RETURN 
  collect(DISTINCT properties(n)) AS n_properties, 
  collect(DISTINCT type(r)) AS r_type, 
  collect(DISTINCT properties(r)) AS r_properties, 
  collect(DISTINCT properties(m)) AS m_properties

The question is:
{question}
"""

CYPHER_QA_TEMPLATE = """
Bạn là trợ lý ảo giúp trả lời các câu hỏi dựa trên thông tin được cung cấp bên dưới.

Yêu cầu:
- Chỉ sử dụng thông tin đã cho, không thêm suy đoán hay kiến thức bên ngoài.
- Trả lời đúng trọng tâm, rõ ràng, tự nhiên và ngắn gọn.
- Nếu thông tin có đề cập đến **ngân hàng, số tài khoản, hình thức thanh toán**, hãy ưu tiên làm nổi bật các chi tiết đó.
- Nếu câu hỏi liên quan đến đối tượng như **sinh viên, giảng viên, môn học, đồ án, học phí…**, hãy trả lời phù hợp với ngữ cảnh.
- Nếu không đủ thông tin để trả lời, hãy lịch sự nói rằng bạn không có thông tin cần thiết.
- Không nhắc lại thông tin nguồn hay nói rằng bạn đang dựa vào dữ liệu đã cho.

Ví dụ:

Câu hỏi: Tôi có thể đóng học phí ở đâu?  
Thông tin: ["ngân hàng": "Ngân hàng A", "số tài khoản": "123456789", "chủ tài khoản": "Trường Đại học X"]  
Câu trả lời: Bạn có thể đóng học phí qua Ngân hàng A, số tài khoản 123456789, chủ tài khoản Trường Đại học X.

Thông tin:
{context}

Câu hỏi: {question}  
Câu trả lời:
"""

CYPHER_QA_PROMPT = PromptTemplate(
  input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)

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
  qa_prompt=CYPHER_QA_PROMPT,
	return_intermediate_steps=True,
	cypher_prompt=CYPHER_GENERATION_PROMPT,
)

def create_app():
    app = Flask(__name__, template_folder='views')

    from app.controllers.chat_controller import chat_bp
    app.register_blueprint(chat_bp)

    return app
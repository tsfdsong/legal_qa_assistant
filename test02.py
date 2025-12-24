# -*- coding: utf-8 -*-
import json
import time
from pathlib import Path
from typing import List, Dict
import numpy as np
import jieba

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings, get_response_synthesizer
from llama_index.core.schema import TextNode
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import PromptTemplate
from llama_index.core.postprocessor import SentenceTransformerRerank  # 新增重排序组件
from llama_index.llms.openai_like import OpenAILike

# QA_TEMPLATE = (
#     "<|im_start|>system\n"
#     "您是中国劳动法领域专业助手，必须严格遵循以下规则：\n"
#     "1.仅使用提供的法律条文回答问题\n"
#     "2.若问题与劳动法无关或超出知识库范围，明确告知无法回答\n"
#     "3.引用条文时标注出处\n\n"
#     "可用法律条文（共{context_count}条）：\n{context_str}\n<|im_end|>\n"
#     "<|im_start|>user\n问题：{query_str}<|im_end|>\n"
#     "<|im_start|>assistant\n"
# )


# response_template = PromptTemplate(QA_TEMPLATE)

# ================== 配置区 ==================
class Config:
    EMBED_MODEL_PATH = r"/home/cw/llms/embedding_model/sungw111/text2vec-base-chinese-sentence"
    RERANK_MODEL_PATH = r"/home/cw/llms/rerank_model/BAAI/bge-reranker-large"  # 新增重排序模型路径
    LLM_MODEL_PATH = r"/home/cw/llms/Qwen/Qwen1.5-1.8B-Chat-law"
    
    DATA_DIR = "./data"
    VECTOR_DB_DIR = "./chroma_db"
    PERSIST_DIR = "./storage"
    
    COLLECTION_NAME = "chinese_labor_laws"
    TOP_K = 30  # 扩大初始检索数量
    RERANK_TOP_K = 10  # 重排序后保留数量

# ================== 初始化模型 ==================
def init_models():
    """初始化模型并验证"""
    # Embedding模型
    embed_model = HuggingFaceEmbedding(
        model_name=Config.EMBED_MODEL_PATH,
    )
    
    # LLM
    # llm = HuggingFaceLLM(
    #     model_name=Config.LLM_MODEL_PATH,
    #     tokenizer_name=Config.LLM_MODEL_PATH,
    #     model_kwargs={
    #         "trust_remote_code": True,
    #     },
    #     tokenizer_kwargs={"trust_remote_code": True},
    #     generate_kwargs={"temperature": 0.3}
    # )

    #openai_like
    llm = OpenAILike(
    model="/home/cw/llms/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    api_base="http://localhost:23333/v1",
    api_key="fake",
    context_window=4096,
    is_chat_model=True,
    is_function_calling_model=False,
    )
    # llm = OpenAILike(
    #     model="glm-4",  # 可选模型：glm-4, glm-3-turbo, characterglm等
    #     api_base="https://open.bigmodel.cn/api/paas/v4",  # 关键！必须指定此端点
    #     api_key="d93f31ff90754cebb43413b3bf4a72d1.tnvl4GOxCF8uHCgt",
    #     context_window=128000,    # 按需调整（glm-4实际支持128K）
    #     is_chat_model=True,
    #     is_function_calling_model=False,  # GLM暂不支持函数调用
    #     max_tokens=1024,          # 最大生成token数（按需调整）
    #     temperature=0.3,          # 推荐范围 0.1~1.0
    #     top_p=0.7                 # 推荐范围 0.5~1.0
    # )
    
    # 初始化重排序器（新增）
    reranker = SentenceTransformerRerank(
        model=Config.RERANK_MODEL_PATH,
        top_n=Config.RERANK_TOP_K
    )
    
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    # 验证模型
    test_embedding = embed_model.get_text_embedding("测试文本")
    print(f"Embedding维度验证：{len(test_embedding)}")
    
    return embed_model, llm, reranker  # 返回重排序器

# ================== 数据处理 ==================
def load_and_validate_json_files(data_dir: str) -> List[Dict]:
    """加载并验证JSON法律文件"""
    json_files = list(Path(data_dir).glob("*.json"))
    assert json_files, f"未找到JSON文件于 {data_dir}"
    
    all_data = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # 验证数据结构
                if not isinstance(data, list):
                    raise ValueError(f"文件 {json_file.name} 根元素应为列表")
                for item in data:
                    if not isinstance(item, dict):
                        raise ValueError(f"文件 {json_file.name} 包含非字典元素")
                    for k, v in item.items():
                        if not isinstance(v, str):
                            raise ValueError(f"文件 {json_file.name} 中键 '{k}' 的值不是字符串")
                all_data.extend({
                    "content": item,
                    "metadata": {"source": json_file.name}
                } for item in data)
            except Exception as e:
                raise RuntimeError(f"加载文件 {json_file} 失败: {str(e)}")
    
    print(f"成功加载 {len(all_data)} 个法律文件条目")
    return all_data

def create_nodes(raw_data: List[Dict]) -> List[TextNode]:
    """添加ID稳定性保障"""
    nodes = []
    for entry in raw_data:
        law_dict = entry["content"]
        source_file = entry["metadata"]["source"]
        
        for full_title, content in law_dict.items():
            # 生成稳定ID（避免重复）
            node_id = f"{source_file}::{full_title}"
            
            parts = full_title.split(" ", 1)
            law_name = parts[0] if len(parts) > 0 else "未知法律"
            article = parts[1] if len(parts) > 1 else "未知条款"
            
            node = TextNode(
                text=content,
                id_=node_id,  # 显式设置稳定ID
                metadata={
                    "law_name": law_name,
                    "article": article,
                    "full_title": full_title,
                    "source_file": source_file,
                    "content_type": "legal_article"
                }
            )
            nodes.append(node)
    
    print(f"生成 {len(nodes)} 个文本节点（ID示例：{nodes[0].id_}）")
    return nodes

# ================== 向量存储 ==================

def init_vector_store(nodes: List[TextNode]) -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)
    chroma_collection = chroma_client.get_or_create_collection(
        name=Config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 确保存储上下文正确初始化
    storage_context = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
    )

    # 判断是否需要新建索引
    if chroma_collection.count() == 0 and nodes is not None:
        print(f"创建新索引（{len(nodes)}个节点）...")
        
        # 显式将节点添加到存储上下文
        storage_context.docstore.add_documents(nodes)  
        
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True
        )
        # 双重持久化保障
        storage_context.persist(persist_dir=Config.PERSIST_DIR)
        index.storage_context.persist(persist_dir=Config.PERSIST_DIR)  # <-- 新增
    else:
        print("加载已有索引...")
        storage_context = StorageContext.from_defaults(
            persist_dir=Config.PERSIST_DIR,
            vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
        )
        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )

    # 安全验证
    print("\n存储验证结果：")
    doc_count = len(storage_context.docstore.docs)
    print(f"DocStore记录数：{doc_count}")
    
    if doc_count > 0:
        sample_key = next(iter(storage_context.docstore.docs.keys()))
        print(f"示例节点ID：{sample_key}")
    else:
        print("警告：文档存储为空，请检查节点添加逻辑！")
    
    
    return index

#新增过滤函数
def is_legal_question(text: str) -> bool:
    """判断问题是否属于法律咨询"""
    legal_keywords = ["劳动法", "合同", "工资", "工伤", "解除", "赔偿"]
    return any(keyword in text for keyword in legal_keywords)

# ================== 新增评估类 ==================
class RecallEvaluator:
    def __init__(self, retriever, reranker):
        self.retriever = retriever
        self.reranker = reranker

    def calculate_recall(self, retrieved_nodes, relevant_ids):
        retrieved_ids = [n.node.metadata["full_title"] for n in retrieved_nodes]
        hit = len(set(retrieved_ids) & set(relevant_ids))
        return hit / len(relevant_ids) if relevant_ids else 0.0

    def evaluate(self, benchmark):
        results = []
        for case in benchmark:
            # 初始检索
            initial_nodes = self.retriever.retrieve(case["question"])
            # 重排序
            reranked_nodes = self.reranker.postprocess_nodes(
                initial_nodes, 
                query_str=case["question"]
            )
            # 计算召回率
            recall = self.calculate_recall(reranked_nodes, case["relevant_ids"])
            results.append(recall)
            
            print(f"问题：{case['question']}")
            print(f"初始检索结果：{[n.node.metadata['full_title'] for n in initial_nodes]}")
            print(f"重排序后结果：{[n.node.metadata['full_title'] for n in reranked_nodes]}")
            print(f"召回条款：{[n.node.metadata['full_title'] for n in reranked_nodes[:3]]}")
            print(f"目标条款：{case['relevant_ids']}")
            print(f"召回率：{recall:.1%}\n")
        
        avg_recall = np.mean(results)
        print(f"平均召回率：{avg_recall:.1%}")
        return avg_recall

class E2EEvaluator:
    def __init__(self, query_engine):
        self.query_engine = query_engine

    def evaluate_case(self, response, standard):
        try:
            # 获取实际命中的条款
            retrieved_clauses = [node.node.metadata["full_title"] for node in response.source_nodes]
            
            # 获取标准答案要求的条款
            required_clauses = standard["standard_answer"]["条款"]
            
            # 计算命中情况
            hit_clauses = list(set(retrieved_clauses) & set(required_clauses))
            missed_clauses = list(set(required_clauses) - set(retrieved_clauses))
            
            # 计算命中率
            clause_hit = len(hit_clauses) / len(required_clauses) if required_clauses else 0.0
            
            return {
                "clause_score": clause_hit,
                "hit_clauses": hit_clauses,
                "missed_clauses": missed_clauses
            }
        except Exception as e:
            print(f"评估失败：{str(e)}")
            return None

    def evaluate(self, benchmark):
        results = []
        for case in benchmark:
            try:
                response = self.query_engine.query(case["question"])
                case_result = self.evaluate_case(response, case)
                
                if case_result:
                    print(f"\n问题：{case['question']}")
                    print(f"命中条款：{case_result['hit_clauses']}")
                    print(f"缺失条款：{case_result['missed_clauses']}")
                    print(f"条款命中率：{case_result['clause_score']:.1%}")
                    results.append(case_result)
                else:
                    results.append(None)
            except Exception as e:
                print(f"查询失败：{str(e)}")
                results.append(None)
        
        # 计算统计数据
        valid_results = [r for r in results if r is not None]
        avg_hit = np.mean([r["clause_score"] for r in valid_results]) if valid_results else 0
        
        print("\n=== 最终评估报告 ===")
        print(f"有效评估案例：{len(valid_results)}/{len(benchmark)}")
        print(f"平均条款命中率：{avg_hit:.1%}")
        
        # 输出详细错误分析
        for i, result in enumerate(results):
            if result is None:
                print(f"案例{i+1}：{benchmark[i]['question']} 评估失败")
        
        return results

# ================== 新增测试数据集 ==================
RETRIEVAL_BENCHMARK = [
    # 劳动合同解除类
    {
        "question": "劳动者可以立即解除劳动合同的情形有哪些？",
        "relevant_ids": ["中华人民共和国劳动合同法 第三十八条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第三十九条", "中华人民共和国劳动法 第三十二条"]
    },
    {
        "question": "用人单位单方解除劳动合同需要提前多久通知？",
        "relevant_ids": ["中华人民共和国劳动合同法 第四十条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第三十七条", "中华人民共和国劳动法 第二十六条"]
    },
    
    # 工资与补偿类
    {
        "question": "经济补偿金的计算标准是什么？",
        "relevant_ids": ["中华人民共和国劳动合同法 第四十七条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第八十七条", "中华人民共和国劳动法 第二十八条"]
    },
    {
        "question": "试用期工资最低标准是多少？",
        "relevant_ids": ["中华人民共和国劳动合同法 第二十条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第十九条", "中华人民共和国劳动法 第四十八条"]
    },
    
    # 工伤与福利类
    {
        "question": "工伤认定需要哪些材料？",
        "relevant_ids": ["中华人民共和国劳动合同法 第三十条"],
        "confusing_ids": ["中华人民共和国劳动法 第七十三条", "中华人民共和国劳动合同法 第十七条"]
    },
    {
        "question": "女职工产假有多少天？",
        "relevant_ids": ["中华人民共和国劳动法 第六十二条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第四十二条", "中华人民共和国劳动法 第六十一条"]
    },
    
    # 劳动合同订立类
    {
        "question": "无固定期限劳动合同的订立条件是什么？",
        "relevant_ids": ["中华人民共和国劳动合同法 第十四条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第十三条", "中华人民共和国劳动法 第二十条"]
    },
    {
        "question": "劳动合同必须包含哪些条款？",
        "relevant_ids": ["中华人民共和国劳动合同法 第十七条"],
        "confusing_ids": ["中华人民共和国劳动法 第十九条", "中华人民共和国劳动合同法 第十条"]
    },
    
    # 特殊用工类
    {
        "question": "劳务派遣岗位的限制条件是什么？",
        "relevant_ids": ["中华人民共和国劳动合同法 第六十六条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第五十八条", "中华人民共和国劳动法 第二十条"]
    },
    {
        "question": "非全日制用工的每日工作时间上限？",
        "relevant_ids": ["中华人民共和国劳动合同法 第六十八条"],
        "confusing_ids": ["中华人民共和国劳动法 第三十六条", "中华人民共和国劳动合同法 第三十八条"]
    },
    # 劳动合同终止类
    {
        "question": "劳动合同终止的法定情形有哪些？",
        "relevant_ids": ["中华人民共和国劳动合同法 第四十四条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第四十六条", "中华人民共和国劳动法 第二十三条"]
    },
    {
        "question": "劳动合同期满后必须续签的情形？",
        "relevant_ids": ["中华人民共和国劳动合同法 第四十五条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第十四条", "中华人民共和国劳动法 第二十条"]
    },
    
    # 劳动保护类
    {
        "question": "女职工哺乳期工作时间限制？",
        "relevant_ids": ["中华人民共和国劳动法 第六十三条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第四十二条", "中华人民共和国劳动法 第六十一条"]
    },
    {
        "question": "未成年工禁止从事的劳动类型？",
        "relevant_ids": ["中华人民共和国劳动法 第六十四条"],
        "confusing_ids": ["中华人民共和国劳动法 第五十九条", "中华人民共和国劳动合同法 第六十六条"]
    },
    
    {
        "question": "工伤保险待遇包含哪些项目？",
        "relevant_ids": ["中华人民共和国劳动法 第七十三条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第三十条", "中华人民共和国劳动法 第四十四条"]
    },
    
    # 劳动争议类
    {
        "question": "劳动争议仲裁时效是多久？",
        "relevant_ids": ["中华人民共和国劳动法 第八十二条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第六十条", "中华人民共和国劳动法 第七十九条"]
    },
    {
        "question": "集体合同的法律效力如何？",
        "relevant_ids": ["中华人民共和国劳动法 第三十五条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第五十五条", "中华人民共和国劳动法 第三十三条"]
    },
    
    # 特殊条款类
    {
        "question": "服务期违约金的上限规定？",
        "relevant_ids": ["中华人民共和国劳动合同法 第二十二条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第二十三条", "中华人民共和国劳动法 第一百零二条"]
    },
    {
        "question": "无效劳动合同的认定标准？",
        "relevant_ids": ["中华人民共和国劳动合同法 第二十六条"],
        "confusing_ids": ["中华人民共和国劳动法 第十八条", "中华人民共和国劳动合同法 第三十九条"]
    }
]

E2E_BENCHMARK = [
    # 案例1：劳动合同解除
    {
        "question": "用人单位在哪些情况下不得解除劳动合同？",
        "standard_answer": {
            "条款": ["中华人民共和国劳动合同法 第四十二条"],
            "标准答案": "根据《劳动合同法》第四十二条，用人单位不得解除劳动合同的情形包括：\n1. 从事接触职业病危害作业的劳动者未进行离岗前职业健康检查\n2. 在本单位患职业病或者因工负伤并被确认丧失/部分丧失劳动能力\n3. 患病或非因工负伤在规定的医疗期内\n4. 女职工在孕期、产期、哺乳期\n5. 连续工作满15年且距退休不足5年\n6. 法律、行政法规规定的其他情形\n违法解除需按第八十七条支付二倍经济补偿金",
            "必备条件": ["职业病危害作业未检查", "孕期女职工", "连续工作满15年"]
        }
    },
    
    # 案例2：工资支付
    {
        "question": "拖欠工资劳动者可以采取哪些措施？",
        "standard_answer": {
            "条款": ["中华人民共和国劳动合同法 第三十条", "中华人民共和国劳动法 第五十条"],
            "标准答案": "劳动者可采取以下救济措施：\n1. 根据劳动合同法第三十条向法院申请支付令\n2. 依据劳动合同法第三十八条解除合同并要求经济补偿\n3. 向劳动行政部门投诉\n逾期未支付的，用人单位需按应付金额50%-100%加付赔偿金（劳动合同法第八十五条）",
            "必备条件": ["支付令申请", "解除劳动合同", "行政投诉"]
        }
    },
    
    
    # 案例3：竞业限制
    {
        "question": "竞业限制的最长期限是多久？",
        "standard_answer": {
            "条款": ["中华人民共和国劳动合同法 第二十四条"],
            "标准答案": "根据劳动合同法第二十四条：\n- 竞业限制期限不得超过二年\n- 适用人员限于高管/高级技术人员/其他保密人员\n- 需按月支付经济补偿\n注意区分服务期约定（第二十二条）",
            "限制条件": ["期限≤2年", "按月支付补偿"]
        }
    },
    
    
    # 案例4：劳务派遣
    {
        "question": "劳务派遣用工的比例限制是多少？",
        "standard_answer": {
            "条款": ["中华人民共和国劳动合同法 第六十六条"],
            "标准答案": "劳务派遣用工限制：\n- 临时性岗位不超过6个月\n- 辅助性岗位≤用工总量10%\n- 违法派遣按每人1000-5000元罚款（第九十二条）\n派遣协议需包含岗位/期限/报酬等条款（第五十九条）",
            "限制条件": ["临时性≤6月", "辅助性≤10%"]
        }
    },
    
    # 案例5：非全日制用工
    {
        "question": "非全日制用工的工资支付周期要求？",
        "standard_answer": {
            "条款": ["中华人民共和国劳动合同法 第七十二条"],
            "标准答案": "非全日制用工支付规则：\n- 工资结算周期≤15日\n- 小时工资≥当地最低标准\n- 终止用工不支付经济补偿（第七十一条）\n区别于全日制月薪制（第三十条）",
            "支付规则": ["周期≤15天", "小时工资≥最低标准"]
        }
    },
    
    
    # 案例6：劳动合同无效
    {
        "question": "劳动合同被确认无效后的工资支付标准？",
        "standard_answer": {
            "条款": ["中华人民共和国劳动合同法 第二十八条"],
            "标准答案": "无效劳动合同的工资支付：\n1. 参照本单位相同岗位工资支付\n2. 无相同岗位的按市场价\n3. 已付报酬不足的需补差\n过错方需承担赔偿责任（第八十六条）",
            "支付规则": ["参照同岗位", "市场价补差"]
        }
    }
]

# ================== 主程序 ==================
def main():
    embed_model, llm, reranker = init_models()  # 获取重排序器
    
    # 仅当需要更新数据时执行
    if not Path(Config.VECTOR_DB_DIR).exists():
        print("\n初始化数据...")
        raw_data = load_and_validate_json_files(Config.DATA_DIR)
        nodes = create_nodes(raw_data)
    else:
        nodes = None
    
    print("\n初始化向量存储...")
    start_time = time.time()
    index = init_vector_store(nodes)
    print(f"索引加载耗时：{time.time()-start_time:.2f}s")
    
    # 创建检索器和响应合成器（修改部分）
    retriever = index.as_retriever(
        similarity_top_k=Config.TOP_K  # 扩大初始检索数量
    )
    query_engine = index.as_query_engine(
        similarity_top_k=Config.TOP_K,
        node_postprocessors=[reranker]
    )
    response_synthesizer = get_response_synthesizer(
        # text_qa_template=response_template,
        verbose=True
    )
    # 新增评估模式
    run_mode = input("请选择模式：1-问答模式 2-评估模式\n输入选项：")
    
    if run_mode == "2":
        print("\n=== 开始评估 ===")
        
        # 召回率评估
        recall_evaluator = RecallEvaluator(retriever, reranker)
        recall_result = recall_evaluator.evaluate(RETRIEVAL_BENCHMARK)
        
        # 端到端评估
        e2e_evaluator = E2EEvaluator(query_engine)
        e2e_results = e2e_evaluator.evaluate(E2E_BENCHMARK)
        
        # 生成报告
        print("\n=== 最终评估报告 ===")
        print(f"重排序召回率：{recall_result:.1%}")
        print(f"端到端条款命中率：{np.mean([r['clause_score'] for r in e2e_results]):.1%}")
        return
    
    # 示例查询
    while True:
        question = input("\n请输入劳动法相关问题（输入q退出）: ")
        if question.lower() == 'q':
            break
        # 添加问答类型判断（关键修改）
        # if not is_legal_question(question):  # 新增判断函数
        #     print("\n您好！我是劳动法咨询助手，专注解答《劳动法》《劳动合同法》等相关问题。")
        #     continue
       # 执行检索-重排序-过滤-回答流程
        start_time = time.time()
        
        # 1. 初始检索
        initial_nodes = retriever.retrieve(question)
        retrieval_time = time.time() - start_time
        
        # 2. 重排序
        reranked_nodes = reranker.postprocess_nodes(
            initial_nodes, 
            query_str=question
        )
        rerank_time = time.time() - start_time - retrieval_time

        
        # ★★★★★ 添加过滤逻辑在此处 ★★★★★
        
        MIN_RERANK_SCORE = 0.4
        
        # 执行过滤
        filtered_nodes = [
            node for node in reranked_nodes 
            if node.score > MIN_RERANK_SCORE
        ]
        # for node in reranked_nodes:
        #     print(node.score)
        #一般对模型的回复做限制就从filtered_nodes的返回值下手
        print("原始分数样例：",[node.score for node in reranked_nodes[:3]])
        print("重排序过滤后的结果：",filtered_nodes)
        # 空结果处理
        if not filtered_nodes:
            print("你的问题未匹配到相关资料！")
            continue
        # 3. 合成答案（使用过滤后的节点）
        response = response_synthesizer.synthesize(
            question, 
            nodes=filtered_nodes  # 使用过滤后的节点
        )
        synthesis_time = time.time() - start_time - retrieval_time - rerank_time
        
        # 显示结果（修改显示逻辑）
        print(f"\n智能助手回答：\n{response.response}")
        print("\n支持依据：")
        for idx, node in enumerate(reranked_nodes, 1):
            # 兼容新版API的分数获取方式
            initial_score = node.metadata.get('initial_score', node.score)  # 获取初始分数
            rerank_score = node.score  # 重排序后的分数
        
            meta = node.node.metadata
            print(f"\n[{idx}] {meta['full_title']}")
            print(f"  来源文件：{meta['source_file']}")
            print(f"  法律名称：{meta['law_name']}")
            print(f"  初始相关度：{node.node.metadata.get('initial_score', 0):.4f}")  # 安全访问
            print(f"  重排序得分：{getattr(node, 'score', 0):.4f}")  # 兼容属性访问
            print(f"  条款内容：{node.node.text[:100]}...")
        
        print(f"\n[性能分析] 检索: {retrieval_time:.2f}s | 重排序: {rerank_time:.2f}s | 合成: {synthesis_time:.2f}s")

if __name__ == "__main__":
    main()
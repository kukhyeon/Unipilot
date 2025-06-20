#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 evaluation.py를 활용한 프레임워크 비교
"""

import time
from datetime import datetime

# LangChain
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.schema import Document

# LlamaIndex  
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.schema import Document as LlamaDocument
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 기존 evaluation 모듈
from evaluation import ClaudeEvaluationExporter


class QwenLangChainLLM(LLM):
    """Qwen을 LangChain LLM으로 래핑"""
    def __init__(self, model_service):
        super().__init__()
        self.model_service = model_service
    
    def _call(self, prompt: str, stop=None) -> str:
        messages = [{"role": "user", "content": prompt}]
        response, _ = self.model_service.generate_response(messages)
        return response
    
    @property
    def _llm_type(self) -> str:
        return "qwen"


class QwenLlamaIndexLLM(CustomLLM):
    """Qwen을 LlamaIndex LLM으로 래핑"""
    def __init__(self, model_service):
        super().__init__()
        self.model_service = model_service
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=32768, num_output=512, model_name="qwen2.5-14b")
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        messages = [{"role": "user", "content": prompt}]
        response, _ = self.model_service.generate_response(messages)
        return CompletionResponse(text=response)


class LangChainService:
    """LangChain 기반 서비스 (기존 인터페이스와 동일)"""
    def __init__(self, model_service, document_manager):
        self.model_service = model_service
        self.qa_chain = self._setup_langchain(model_service, document_manager)
        
    def _setup_langchain(self, model_service, document_manager):
        # 임베딩 설정
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # 문서 로드
        rag_data = document_manager.load_rag_dataset()
        documents = [Document(page_content=item.get('content', '')) for item in rag_data]
        
        # 벡터스토어 생성
        vectorstore = Chroma.from_documents(documents, embeddings)
        
        # LLM 및 체인 생성
        llm = QwenLangChainLLM(model_service)
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
        )
    
    def ask_question(self, question):
        """기존 인터페이스와 동일"""
        try:
            return self.qa_chain.run(question)
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def get_system_status(self):
        return {"model_loaded": True, "framework": "LangChain"}
    
    def cleanup(self):
        pass


class LlamaIndexService:
    """LlamaIndex 기반 서비스 (기존 인터페이스와 동일)"""
    def __init__(self, model_service, document_manager):
        self.model_service = model_service
        self.query_engine = self._setup_llamaindex(model_service, document_manager)
        
    def _setup_llamaindex(self, model_service, document_manager):
        # 임베딩 및 LLM 설정
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        llm = QwenLlamaIndexLLM(model_service)
        
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        # 문서 로드
        rag_data = document_manager.load_rag_dataset()
        documents = [LlamaDocument(text=item.get('content', '')) for item in rag_data]
        
        # 인덱스 생성
        index = VectorStoreIndex.from_documents(documents)
        return index.as_query_engine(similarity_top_k=5)
    
    def ask_question(self, question):
        """기존 인터페이스와 동일"""
        try:
            response = self.query_engine.query(f"한국어로 답변해주세요: {question}")
            return str(response)
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def get_system_status(self):
        return {"model_loaded": True, "framework": "LlamaIndex"}
    
    def cleanup(self):
        pass


def run_framework_comparison():
    """세 프레임워크 비교 실행"""
    print("🎯 RAG 프레임워크 비교 평가 시작")
    
    # 기존 서비스 로드
    from modules import create_academic_service
    from modules.document_manager import DocumentManager
    
    original_service = create_academic_service()
    document_manager = DocumentManager()
    
    results = {}
    
    # 1. Custom RAG 평가
    print("\n🎯 Custom RAG 평가...")
    evaluator = ClaudeEvaluationExporter()
    evaluator.output_file = f"custom_rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    results['custom'] = evaluator.run_comprehensive_evaluation(original_service, 4)
    
    # 2. LangChain RAG 평가
    print("\n🦜 LangChain RAG 평가...")
    langchain_service = LangChainService(original_service.model_service, document_manager)
    evaluator.output_file = f"langchain_rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    results['langchain'] = evaluator.run_comprehensive_evaluation(langchain_service, 4)
    
    # 3. LlamaIndex RAG 평가
    print("\n🦙 LlamaIndex RAG 평가...")
    llamaindex_service = LlamaIndexService(original_service.model_service, document_manager)
    evaluator.output_file = f"llamaindex_rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    results['llamaindex'] = evaluator.run_comprehensive_evaluation(llamaindex_service, 4)
    
    # 정리
    original_service.cleanup()
    
    # 요약 생성
    summary_file = f"framework_comparison_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"""
RAG 프레임워크 아키텍처 비교 평가 완료
=======================================

실행 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

생성된 파일:
- Custom RAG: {results['custom']}
- LangChain RAG: {results['langchain']}  
- LlamaIndex RAG: {results['llamaindex']}

각 파일을 Claude에게 업로드하여 RAGAS 평가를 요청하세요.
""")
    
    print(f"\n🎉 비교 평가 완료!")
    print(f"📊 요약 파일: {summary_file}")
    for framework, file in results.items():
        print(f"📁 {framework}: {file}")
    
    return results


if __name__ == "__main__":
    run_framework_comparison()

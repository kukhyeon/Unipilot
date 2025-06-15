#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
문서 관리 모듈
문서 로딩, 임베딩, 저장을 담당하며 나중에 계층별 문서 구조로 확장 가능
"""

import json
import os
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer


class DocumentManager:
    """문서 관리 클래스"""
    
    def __init__(self, embedder: Optional[SentenceTransformer] = None):
        self.embedder = embedder
        self.documents = []
        self.metadata = []
        self.embeddings = []
    
    def set_embedder(self, embedder: SentenceTransformer) -> None:
        """임베딩 모델 설정"""
        self.embedder = embedder
    
    def load_json_documents(self, file_paths: List[str]) -> Tuple[List[str], List[Dict]]:
        """JSON 파일들로부터 문서 로딩"""
        documents = []
        metadata = []
        
        for file_path in file_paths:
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # 데이터가 리스트인 경우 (예: 교과목 통합.json)
                    if isinstance(data, list):
                        for i, item in enumerate(data):
                            if isinstance(item, dict):
                                content = item.get("content", "")
                                title = item.get("title", f"항목 {i+1}")
                                
                                if content:
                                    documents.append(content)
                                    metadata.append({
                                        "file": os.path.basename(file_path),
                                        "title": title,
                                        "source": "json_file",
                                        "index": i
                                    })
                    
                    # 데이터가 딕셔너리인 경우 (기존 방식)
                    elif isinstance(data, dict):
                        content = data.get("content", "").strip()
                        
                        if content:
                            documents.append(content)
                            metadata.append({
                                "file": os.path.basename(file_path),
                                "title": data.get("title", ""),
                                "source": "json_file"
                            })
                    
                    else:
                        print(f"⚠️ 지원하지 않는 JSON 형식: {file_path}")
                        
            except Exception as e:
                print(f"⚠️ 파일 로딩 실패: {file_path}, 오류: {e}")
                continue
        
        return documents, metadata
    
    def load_rag_dataset(self, rag_dir: str = "C:/Users/user/Desktop/DeepLearning/LLM/rag_dataset", 
                        extra_files: List[str] = None) -> bool:
        """RAG 데이터셋 로딩"""
        file_paths = []
        
        # 디렉토리 내 JSON 파일들
        if os.path.exists(rag_dir):
            for filename in os.listdir(rag_dir):
                if filename.endswith(".json"):
                    file_paths.append(os.path.join(rag_dir, filename))
        else:
            print(f"⚠️ RAG 디렉토리를 찾을 수 없습니다: {rag_dir}")
        
        # 추가 파일들
        if extra_files:
            for file_path in extra_files:
                if os.path.exists(file_path):
                    file_paths.append(file_path)
        
        # 문서 로딩
        documents, metadata = self.load_json_documents(file_paths)
        
        if documents:
            self.documents = documents
            self.metadata = metadata
            
            # 임베딩 생성
            if self.embedder:
                self.generate_embeddings()
                
            print(f"✅ RAG 데이터셋 로딩 완료: {len(documents)}개 문서")
            return True
        else:
            print("❌ 로딩된 문서가 없습니다.")
            return False
    
    def generate_embeddings(self) -> None:
        """문서들의 임베딩 생성"""
        if not self.embedder:
            raise ValueError("임베딩 모델이 설정되지 않았습니다.")
        
        if self.documents:
            print(f"🔄 {len(self.documents)}개 문서의 임베딩 생성 중...")
            self.embeddings = self.embedder.encode(self.documents, convert_to_numpy=True)
            print("✅ 임베딩 생성 완료")
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None, 
                     generate_embeddings: bool = True) -> None:
        """새 문서 추가"""
        self.documents.extend(documents)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{"source": "manual_add"}] * len(documents))
        
        # 임베딩 재생성
        if generate_embeddings and self.embedder:
            self.generate_embeddings()
        
        print(f"📝 {len(documents)}개 문서가 추가되었습니다.")
    
    def add_transcript_documents(self, transcript_docs: List[Dict]) -> None:
        """성적표 관련 문서들 추가"""
        documents = []
        metadata = []
        
        for doc in transcript_docs:
            documents.append(doc["content"])
            metadata.append({
                "file": "transcript_summary",
                "title": doc["title"],
                "type": doc["type"],
                "source": "transcript"
            })
        
        self.add_documents(documents, metadata)
        print(f"📊 성적표 요약 정보 {len(transcript_docs)}개 문서가 RAG 데이터베이스에 추가되었습니다.")
    
    def get_documents(self) -> Tuple[List[str], List[Dict]]:
        """현재 로딩된 문서들 반환"""
        return self.documents.copy(), self.metadata.copy()
    
    def get_embeddings(self):
        """현재 임베딩들 반환"""
        return self.embeddings
    
    def clear_documents(self) -> None:
        """모든 문서 및 임베딩 초기화"""
        self.documents = []
        self.metadata = []
        self.embeddings = []
        print("🗑️ 모든 문서가 초기화되었습니다.")
    
    def get_document_info(self) -> Dict:
        """문서 현황 정보 반환"""
        sources = {}
        for meta in self.metadata:
            source = meta.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "total_metadata": len(self.metadata),
            "has_embeddings": len(self.embeddings) > 0,
            "embedding_dim": self.embeddings.shape[1] if len(self.embeddings) > 0 else 0,
            "sources": sources
        }
    
    def save_documents_to_json(self, output_path: str) -> bool:
        """문서들을 JSON 파일로 저장"""
        try:
            data = {
                "documents": self.documents,
                "metadata": self.metadata,
                "info": self.get_document_info()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 문서들이 저장되었습니다: {output_path}")
            return True
        except Exception as e:
            print(f"❌ 문서 저장 실패: {e}")
            return False
    
    def load_documents_from_json(self, input_path: str) -> bool:
        """JSON 파일로부터 문서들 로딩"""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.documents = data.get("documents", [])
            self.metadata = data.get("metadata", [])
            
            # 임베딩 재생성
            if self.embedder and self.documents:
                self.generate_embeddings()
            
            print(f"📂 문서들이 로딩되었습니다: {input_path}")
            print(f"📊 로딩된 문서 수: {len(self.documents)}")
            return True
        except Exception as e:
            print(f"❌ 문서 로딩 실패: {e}")
            return False


# 팩토리 함수
def create_document_manager(embedder: Optional[SentenceTransformer] = None) -> DocumentManager:
    """문서 관리자 생성"""
    return DocumentManager(embedder) 
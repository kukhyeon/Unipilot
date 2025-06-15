#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
검색 인터페이스 모듈
검색 전략을 추상화하여 나중에 계층화 검색으로 교체 가능하도록 설계
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SearchStrategy(ABC):
    """검색 전략 추상 클래스"""
    
    @abstractmethod
    def search(self, query: str, k: int = 5, **kwargs) -> List[str]:
        """검색 수행"""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[str], metadata: List[Dict] = None) -> None:
        """문서 추가"""
        pass


class SemanticSearchStrategy(SearchStrategy):
    """의미 기반 검색 전략"""
    
    def __init__(self, embedder):
        self.embedder = embedder
        self.documents = []
        self.metadata = []
        self.embeddings = []
    
    def search(self, query: str, k: int = 5, max_doc_length: int = 1500) -> List[str]:
        """의미 기반 문서 검색"""
        if len(self.embeddings) == 0:
            return []
        
        # 쿼리 임베딩
        q_vec = self.embedder.encode([query], convert_to_numpy=True)
        
        # 유사도 계산
        similarities = cosine_similarity(q_vec, self.embeddings)[0]
        
        # 상위 k개 문서 선택
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # 문서 길이 제한 적용
        limited_docs = []
        for i in top_indices:
            doc = self.documents[i]
            if len(doc) > max_doc_length:
                doc = doc[:max_doc_length] + "..."
            limited_docs.append(doc)
        
        return limited_docs
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None) -> None:
        """문서 추가 및 임베딩 생성"""
        self.documents.extend(documents)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))
        
        # 임베딩 재계산
        if self.documents:
            self.embeddings = self.embedder.encode(self.documents, convert_to_numpy=True)
    
    def clear(self) -> None:
        """모든 문서 및 임베딩 초기화"""
        self.documents = []
        self.metadata = []
        self.embeddings = []


class HierarchicalSearchStrategy(SearchStrategy):
    """계층화 검색 전략 (향후 구현용)"""
    
    def __init__(self, embedder):
        self.embedder = embedder
        # 향후 계층별 검색 로직 구현
        pass
    
    def search(self, query: str, k: int = 5, **kwargs) -> List[str]:
        """계층화 검색 (향후 구현)"""
        # TODO: 계층별 검색 로직 구현
        raise NotImplementedError("계층화 검색은 향후 구현 예정")
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None) -> None:
        """문서 추가 (향후 구현)"""
        # TODO: 계층별 문서 관리 로직 구현
        raise NotImplementedError("계층화 문서 관리는 향후 구현 예정")


class RetrievalInterface:
    """검색 인터페이스 메인 클래스"""
    
    def __init__(self, strategy: SearchStrategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: SearchStrategy) -> None:
        """검색 전략 변경"""
        self.strategy = strategy
    
    def search_relevant_docs(self, query: str, k: int = 5, **kwargs) -> List[str]:
        """관련 문서 검색"""
        return self.strategy.search(query, k, **kwargs)
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None) -> None:
        """문서 추가"""
        self.strategy.add_documents(documents, metadata)
    
    def clear_documents(self) -> None:
        """문서 초기화 (지원하는 전략에 한해)"""
        if hasattr(self.strategy, 'clear'):
            self.strategy.clear()


# 기본 검색 인터페이스 인스턴스 생성용 팩토리 함수
def create_semantic_retrieval_interface(embedder) -> RetrievalInterface:
    """의미 기반 검색 인터페이스 생성"""
    strategy = SemanticSearchStrategy(embedder)
    return RetrievalInterface(strategy)


def create_hierarchical_retrieval_interface(embedder) -> RetrievalInterface:
    """계층화 검색 인터페이스 생성 (향후 구현)"""
    strategy = HierarchicalSearchStrategy(embedder)
    return RetrievalInterface(strategy) 
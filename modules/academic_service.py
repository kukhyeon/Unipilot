#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
학사 상담 서비스 모듈
기본 RAG 시스템을 사용한 한국어 학사 상담 서비스
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

# 모듈 import
from .model_service import ModelService
from .document_manager import DocumentManager
from .retrieval_interface import create_semantic_retrieval_interface
from .context_builder import create_context_builder
from .academic_analyzer import AcademicAnalyzer
from .transcript_summarizer import TranscriptSummarizer

class AcademicCounselingService:
    """학사 상담 서비스 메인 클래스"""
    
    def __init__(self, model_path: str = None):
        """
        학사 상담 서비스 초기화
        
        Args:
            model_path: 모델 경로 (기본값: None, 자동 감지)
        """
        print("🎓 학사 상담 서비스 초기화 중...")
        
        # 기본 설정
        if model_path is None:
            model_path = "C:/Users/user/Desktop/DeepLearning/LLM/Qwen2.5-14B-Instruct"
        self.model_path = model_path
        self.transcript_data = None
        self.analysis_result = None
        
        # 핵심 컴포넌트 초기화
        self.model_service = None
        self.document_manager = None
        self.retrieval_interface = None
        self.context_builder = None
        self.academic_analyzer = None
        self.summarizer = None
        
        # 초기화 수행
        self._initialize_components()
        
        print("학사 상담 서비스 초기화 완료!")
    
    def _initialize_components(self):
        """핵심 컴포넌트들 초기화"""
        try:
            # 1. 모델 서비스 초기화
            print("모델 서비스 초기화 중...")
            self.model_service = ModelService(self.model_path)
            
            # 모델 로딩
            if not self.model_service.load_model():
                raise RuntimeError("모델 로딩 실패")
            
            # 2. 임베딩 모델 초기화
            print("임베딩 모델 로딩 중...")
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            if hasattr(self.model_service, 'device') and self.model_service.device == "cuda":
                self.embedder = self.embedder.to("cuda")
                print("임베딩 모델을 CUDA로 이동 완료")
            
            # 3. 문서 관리자 초기화
            print("문서 관리자 초기화 중...")
            self.document_manager = DocumentManager()
            
            # 4. 검색 인터페이스 초기화 (임베딩 모델 전달)
            print("검색 인터페이스 초기화 중...")
            self.retrieval_interface = create_semantic_retrieval_interface(self.embedder)
            
            # 5. 컨텍스트 빌더 초기화
            print("컨텍스트 빌더 초기화 중...")
            self.context_builder = create_context_builder()
            
            # 6. 학사 분석기 초기화
            print("학사 분석기 초기화 중...")
            self.academic_analyzer = AcademicAnalyzer()
            
            # 7. 성적표 요약기 초기화
            print("성적표 요약기 초기화 중...")
            self.summarizer = TranscriptSummarizer()
            
            # 8. RAG 데이터셋 로딩
            print("RAG 데이터셋 로딩 중...")
            self._load_rag_dataset()
            
            # 9. 성적표 데이터 로드
            self._load_transcript_data()
            
        except Exception as e:
            print(f"컴포넌트 초기화 실패: {e}")
            raise
    
    def _load_rag_dataset(self):
        """RAG 데이터셋 로딩"""
        try:
            rag_dir = "C:/Users/user/Desktop/DeepLearning/LLM/rag_dataset"  
            if os.path.exists(rag_dir):
                # 문서 관리자를 통해 RAG 데이터셋 로딩
                success = self.document_manager.load_rag_dataset(rag_dir)
                
                if success:
                    # 검색 인터페이스에 문서 추가
                    documents, metadata = self.document_manager.get_documents()
                    self.retrieval_interface.add_documents(documents, metadata)
                    print(f"RAG 데이터셋 로딩 완료: {len(documents)}개 문서")
                else:
                    print("RAG 데이터셋 로딩 실패")
            else:
                print(f"RAG 데이터셋 폴더를 찾을 수 없습니다: {rag_dir}")
        except Exception as e:
            print(f"RAG 데이터셋 로딩 중 오류: {e}")
    
    def _load_transcript_data(self):
        """성적표 데이터 로드"""
        try:
            # 원래 경로로 복구
            transcript_path = Path("C:/Users/user/Desktop/DeepLearning/LLM/Project_AI/outputs/transcripts_100/12190002_b6d48c.json")
            if transcript_path.exists():
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    self.transcript_data = json.load(f)
                print("성적표 데이터 로드 완료")
                
                # 학사 분석 수행
                self.analysis_result = self.academic_analyzer.analyze_graduation_requirements(
                    self.transcript_data
                )
                print("졸업 요건 분석 완료")
                
                # 성적표 요약 정보를 RAG에 추가
                self._add_transcript_to_rag()
            else:
                print(f"성적표 데이터 파일을 찾을 수 없습니다: {transcript_path}")
        except Exception as e:
            print(f"성적표 데이터 로드 실패: {e}")
    
    def _add_transcript_to_rag(self):
        """성적표 요약 정보를 RAG에 추가"""
        try:
            if self.transcript_data:
                # 성적표 요약 문서 생성
                docs = self.summarizer.create_multiple_rag_docs(self.transcript_data)
                
                # 검색 인터페이스에 추가
                documents = [doc["content"] for doc in docs]
                metadata = [{"file": "transcript_summary", "title": doc["title"], 
                           "type": doc["type"], "source": "transcript"} for doc in docs]
                self.retrieval_interface.add_documents(documents, metadata)
                
                print(f"성적표 요약 정보 {len(docs)}개 문서가 RAG에 추가되었습니다.")
        except Exception as e:
            print(f"성적표 RAG 추가 중 오류: {e}")
    
    def get_rag_search_function(self):
        """RAG 검색 함수 반환"""
        def basic_search(query: str, k: int = 5, **kwargs) -> List[str]:
            return self.retrieval_interface.search_relevant_docs(query, k=k, **kwargs)
        
        return basic_search
    
    def ask_question(self, user_question: str) -> str:
        """
        사용자 질문에 대한 답변 생성
        
        Args:
            user_question: 사용자 질문
            
        Returns:
            str: 생성된 답변
        """
        try:
            print(f"질문: {user_question}")
            
            # RAG 검색 함수 가져오기
            rag_search_func = self.get_rag_search_function()
            
            # 컨텍스트 생성
            context = self.context_builder.create_full_context(
                user_question=user_question,
                transcript_data=self.transcript_data,
                rag_search_func=rag_search_func,
                analysis_result=self.analysis_result,
                include_semester_details=False  # 기본 요약만 사용
            )
            
            # 메시지 형식으로 구성
            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": user_question}
            ]
            
            # 모델을 통한 답변 생성
            response, metadata = self.model_service.generate_response(messages)
            
            # 로그 저장
            self.model_service.save_conversation_log(
                user_question=user_question,
                model_response=response,
                metadata=metadata,
                system_prompt=context
            )
            
            return response
            
        except Exception as e:
            error_msg = f"질문 처리 중 오류가 발생했습니다: {e}"
            print(f"{error_msg}")
            return error_msg
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 정보 반환"""
        return {
            "model_loaded": self.model_service is not None,
            "transcript_loaded": self.transcript_data is not None,
            "analysis_completed": self.analysis_result is not None,
            "rag_available": self.retrieval_interface is not None,
            "components_status": {
                "model_service": self.model_service is not None,
                "document_manager": self.document_manager is not None,
                "retrieval_interface": self.retrieval_interface is not None,
                "context_builder": self.context_builder is not None,
                "academic_analyzer": self.academic_analyzer is not None,
                "summarizer": self.summarizer is not None
            }
        }
    
    def reload_transcript_data(self, new_data_path: str = None):
        """성적표 데이터 재로드"""
        try:
            if new_data_path:
                with open(new_data_path, 'r', encoding='utf-8') as f:
                    self.transcript_data = json.load(f)
            else:
                self._load_transcript_data()
            
            # 분석 재수행
            if self.transcript_data:
                self.analysis_result = self.academic_analyzer.analyze_graduation_requirements(
                    self.transcript_data
                )
            
            print("성적표 데이터 재로드 완료")
            
        except Exception as e:
            print(f"성적표 데이터 재로드 실패: {e}")
    
    def cleanup(self):
        """리소스 정리"""
        try:
            if self.model_service:
                self.model_service.cleanup()
            print("학사 상담 서비스 정리 완료")
        except Exception as e:
            print(f"정리 중 오류: {e}")
    
    def validate_response(self, question: str, response: str, context_data: Dict) -> Dict[str, Any]:
        """응답 검증 및 일관성 체크"""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "corrections": []
        }
        
        # 숫자 일관성 체크
        if "평점" in question or "학점" in question:
            numbers = self.extract_numbers_from_text(response)
            if len(numbers) >= 2:
                # 비교 문장에서 숫자 순서 체크
                if "더 좋" in response or "더 높" in response:
                    if numbers[0] < numbers[1] and "첫 번째가 더" in response:
                        validation_result["warnings"].append("숫자 비교 결과가 모순됩니다.")
        
        # 정보 존재 여부 체크
        if "정보를 찾을 수 없" in response or "알 수 없" in response:
            # 실제로 정보가 있는지 확인
            if self.check_info_availability(question, context_data):
                validation_result["warnings"].append("실제로는 정보가 존재할 수 있습니다.")
        
        return validation_result
    
    def extract_numbers_from_text(self, text: str) -> List[float]:
        """텍스트에서 숫자 추출"""
        import re
        numbers = re.findall(r'\d+\.?\d*', text)
        return [float(n) for n in numbers if n]
    
    def check_info_availability(self, question: str, context_data: Dict) -> bool:
        """질문에 대한 정보가 실제로 존재하는지 확인"""
        # 간단한 키워드 기반 체크
        if "학기" in question and "수강" in question:
            # 성적표 데이터에서 해당 학기 정보 확인
            transcript_data = context_data.get("transcript_data", {})
            semesters = transcript_data.get("ground_truth", {}).get("semesters", [])
            return len(semesters) > 0
        
        return False


# 팩토리 함수
def create_academic_service(model_path: str = None) -> AcademicCounselingService:
    """학사 상담 서비스 팩토리 함수"""
    return AcademicCounselingService(model_path)


# 메인 실행부
if __name__ == "__main__":
    # 테스트용 질문들
    from .test_questions import TEST_QUESTIONS
    
    print("학사 상담 서비스 테스트 시작")
    
    # 서비스 초기화
    service = create_academic_service()
    
    # 시스템 상태 확인
    status = service.get_system_status()
    print(f"시스템 상태: {status}")
    
    # 테스트 질문들 실행
    test_questions = TEST_QUESTIONS.get("level_1", [])[:3]  # 처음 3개만 테스트
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*50}")
        print(f"테스트 {i}: {question}")
        print('='*50)
        
        start_time = time.time()
        response = service.ask_question(question)
        processing_time = time.time() - start_time
        
        print(f"처리 시간: {processing_time:.2f}초")
        print(f"응답: {response}")
    
    # 정리
    service.cleanup()
    print("\n테스트 완료") 
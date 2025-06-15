#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모듈화된 Qwen 2.5 14B 기반 학사 상담 시스템
기본 RAG 시스템 사용
"""

import json
import os
import argparse
from modules import create_academic_service
from modules.test_questions import TEST_QUESTIONS


def main():
    """메인 실행 함수"""
    print("="*80)
    print("🎓 Qwen 2.5 14B 기반 학사 상담 시스템")
    print("📊 기본 RAG 시스템 사용")
    print("="*80)
    
    # 학사 상담 서비스 생성
    service = create_academic_service()
    
    # 성적표 파일 경로
    transcript_path = r"C:\Users\user\Desktop\DeepLearning\LLM\Project_AI\outputs\transcripts_100\12190002_b6d48c.json"
    
    try:
        if os.path.exists(transcript_path):
            print("📄 성적표 데이터가 이미 로드되어 있습니다.")
        else:
            print(f"⚠️ 성적표 파일을 찾을 수 없습니다: {transcript_path}")
            print("일반 질문 테스트를 실행합니다.")
        
        # 모든 테스트 질문 실행
        from modules.test_questions import get_all_transcript_questions
        test_questions = get_all_transcript_questions()
        
        print("\n📋 테스트 질문 실행:")
        print("="*60)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n[{i}/{len(test_questions)}] 질문: {question}")
            try:
                response = service.ask_question(question)
                print(f"답변: {response}")
            except Exception as e:
                print(f"❌ 오류: {e}")
            print("-" * 60)
        
        print("\n🎉 테스트 완료!")
        
        # 시스템 상태 출력
        status = service.get_system_status()
        print(f"\n📊 시스템 상태:")
        print(f"  - 모델 로딩: {status['model_loaded']}")
        print(f"  - 성적표 로딩: {status['transcript_loaded']}")
        print(f"  - 분석 완료: {status['analysis_completed']}")
        print(f"  - RAG 사용 가능: {status['rag_available']}")
        
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
    
    finally:
        # 리소스 정리
        service.cleanup()


def interactive_mode():
    """대화형 모드"""
    print("🎯 대화형 모드 시작")
    print("📊 기본 RAG 시스템 사용")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    
    service = create_academic_service()
    
    try:
        while True:
            user_input = input(f"\n❓ 질문을 입력하세요: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료']:
                break
            
            if not user_input:
                continue
            
            print(f"\n🤖 답변 생성 중...")
            try:
                response = service.ask_question(user_input)
                print(f"✅ 답변: {response}")
            except Exception as e:
                print(f"❌ 오류: {e}")
            
    except KeyboardInterrupt:
        print("\n\n👋 대화형 모드 종료")
    
    finally:
        service.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모듈화된 Qwen 2.5 14B 학사 상담 시스템")
    parser.add_argument("--interactive", action="store_true", help="대화형 모드 실행")
    
    args = parser.parse_args()
    
    # python qwen_modular_inference.py
    if args.interactive:
        interactive_mode()
    else:
        main() 
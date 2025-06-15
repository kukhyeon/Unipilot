#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
컨텍스트 구성 모듈
RAG 결과 + 분석 결과를 조합하여 시스템 프롬프트 구성
나중에 계층별 컨텍스트 구성으로 확장 가능
"""

import json
from typing import Dict, List, Any, Optional
from .transcript_summarizer import TranscriptSummarizer


class ContextBuilder:
    """컨텍스트 구성 클래스"""
    
    def __init__(self):
        self.summarizer = TranscriptSummarizer()
    
    def create_improved_system_prompt(self, transcript_summary: str, regulation: str, 
                                    summary: str, rag_context: str = "", analysis_result: Dict = None) -> str:
        """Qwen에 최적화된 시스템 프롬프트 생성 (분석 결과 포함)"""
        
        # regulation JSON을 간단히 요약
        if isinstance(regulation, str):
            try:
                reg_data = json.loads(regulation)
            except:
                reg_data = {}
        else:
            reg_data = regulation
        
        reg_summary = f"""졸업학점: {reg_data.get('graduation_credits', 130)}학점
전공필수: {reg_data.get('major_required_credits', 45)}학점
전공선택: {reg_data.get('major_elective_credits', 30)}학점"""
        
        # 분석 결과 포함
        analysis_section = ""
        if analysis_result and "error" not in analysis_result:
            # 기본 정보
            student_info = analysis_result.get("student_info", {})
            student_name = student_info.get('name', 'N/A')
            if isinstance(student_name, list):
                student_name = student_name[0] if student_name else 'N/A'
            
            basic_info = f"""👤 기본 정보:
- 학번: {student_info.get('id', 'N/A')}
- 이름: {student_name}
- 전공: {student_info.get('major', 'N/A')}"""
            
            # 졸업 분석 결과 (새로운 구조에 맞춤)
            requirements_status = analysis_result.get("requirements_status", {})
            if requirements_status:
                total_credits = requirements_status.get("total_credits", {})
                major_credits = requirements_status.get("major_credits", {})
                capstone = requirements_status.get("capstone", {})
                english_cert = requirements_status.get("english_certification", {})
                gpa_info = requirements_status.get("gpa", {})
                
                # 교양학점 계산 (총학점 - 전공학점)
                general_credits_earned = total_credits.get("completed", 0) - major_credits.get("completed", 0)
                general_credits_required = total_credits.get("required", 130) - major_credits.get("required", 65)
                
                graduation_info = f"""🎓 졸업 요건 분석 결과:
- 총 이수학점: {total_credits.get('completed', 0)}학점 / {total_credits.get('required', 130)}학점 {'✅' if total_credits.get('satisfied', False) else '❌'}
- 전공학점: {major_credits.get('completed', 0)}학점 / {major_credits.get('required', 65)}학점 {'✅' if major_credits.get('satisfied', False) else '❌'}
- 교양학점: {general_credits_earned}학점 / {general_credits_required}학점 {'✅' if general_credits_earned >= general_credits_required else '❌'}
- 종합설계: {'이수완료 ✅' if capstone.get('satisfied', False) else '미이수 ❌'}
- 영어인증: {'완료 ✅' if english_cert.get('satisfied', False) else '미완료 ❌'}
- 평점평균: {gpa_info.get('current', 0.0)} / {gpa_info.get('required', 2.0)} {'✅' if gpa_info.get('satisfied', False) else '❌'}

🎯 졸업 가능 여부: {'가능 ✅' if analysis_result.get('is_graduation_ready', False) else '추가 요건 필요 ❌'}"""
                
                # 미충족 요건
                remaining_requirements = analysis_result.get("remaining_requirements", [])
                if remaining_requirements:
                    unmet_info = "\n📋 미충족 요건:\n" + "\n".join([f"- {req}" for req in remaining_requirements])
                    graduation_info += unmet_info
                
                analysis_section = f"\n{basic_info}\n\n{graduation_info}\n"
        
        system_prompt = f"""당신은 한국의 인하대학교 학사 상담사입니다.

🚨 절대 금지 사항 🚨
- 중국어 사용 절대 금지 (中文绝对禁止)
- 영어 사용 금지 (English prohibited)
- 기타 언어 사용 금지

⚠️ 중요: 반드시 한국어로만 답변하세요! ⚠️

✅ 필수 사항:
- 100% 한국어로만 답변
- "안녕하세요", "네", "습니다" 등 자연스러운 한국어 사용
- 정중하고 친절한 존댓말
- 중국어나 영어가 섞이면 안 됨

📋 답변 템플릿:
"안녕하세요! [학생명]님의 질문에 답변드리겠습니다.
[구체적인 답변 내용]
추가 궁금한 점이 있으시면 언제든 말씀해 주세요."

--- 학생 정보 ---
{transcript_summary}

--- 졸업요건 ---  
{reg_summary}
{analysis_section}
--- 분석결과 ---
{summary}

{rag_context}

⚠️ 다시 강조: 반드시 한국어로만 답변하세요! ⚠️"""
        return system_prompt
    
    def create_basic_system_prompt(self) -> str:
        """기본 시스템 프롬프트 (성적표 데이터가 없는 경우)"""
        return """IMPORTANT: You MUST respond ONLY in Korean language. NO Chinese. NO English.
당신은 반드시 한국어로만 답변해야 합니다. 중국어 사용 금지. 영어 사용 금지.

당신은 한국 대학교의 한국인 학사 상담사입니다.
모든 답변은 반드시 한국어로만 작성하세요.

절대 금지사항:
- 중국어 사용 금지 (禁止使用中文)
- 영어 사용 금지 (No English allowed)

중요 규칙:
1. 반드시 한국어로만 답변하세요
2. 정확한 정보만 제공하세요  
3. 학생에게 도움이 되는 조언을 한국어로 해주세요

학생의 질문에 친근하고 정확하게 한국어로 답변해주세요."""
    
    def create_rag_context(self, relevant_docs: List[str]) -> str:
        """RAG 검색 결과로부터 컨텍스트 생성"""
        if not relevant_docs:
            return ""
        
        semantic_context = "\n---\n".join(relevant_docs)
        return f"관련정보:\n{semantic_context}\n\n"
    
    def create_transcript_summary_context(self, transcript_data: Dict, 
                                        include_semester_details: bool = True,
                                        max_length: int = 3000) -> str:
        """성적표 요약 컨텍스트 생성"""
        # 기본 요약 정보
        basic_summary = self.summarizer.create_rag_document(transcript_data)
        
        if not include_semester_details:
            return basic_summary
        
        # 학기별 상세 정보 추가
        semester_details = self.create_semester_details(transcript_data)
        
        # 전체 컨텍스트 구성
        full_context = f"{basic_summary}\n\n{semester_details}"
        
        # 길이 제한 적용 (더 관대하게)
        if len(full_context) > max_length:
            print(f"⚠️ 성적표 컨텍스트가 {len(full_context)}자로 길어서 {max_length}자로 축약합니다.")
            # 기본 요약은 유지하고 학기별 정보만 축약
            if len(basic_summary) < max_length * 0.4:
                remaining_length = max_length - len(basic_summary) - 100
                semester_details = semester_details[:remaining_length] + "\n[학기별 정보가 길이 제한으로 축약됨]"
                full_context = f"{basic_summary}\n\n{semester_details}"
            else:
                full_context = basic_summary[:max_length] + "\n[기본 요약이 길이 제한으로 축약됨]"
        
        return full_context
    
    def create_semester_details(self, transcript_data: Dict) -> str:
        """학기별 상세 과목 정보 생성"""
        semesters = transcript_data.get("ground_truth", {}).get("semesters", [])
        
        details = ["📅 학기별 상세 수강 과목:"]
        
        for semester in semesters:
            year_term = semester.get("year_term", "")
            subjects = semester.get("subjects", [])
            earned = semester.get("earned", 0)
            gpa = semester.get("gpa", 0.0)
            
            details.append(f"\n🔸 {year_term} ({earned}학점, 평점 {gpa}):")
            
            # 과목별 정보
            for subject in subjects:
                name = subject.get("name", "")
                course_number = subject.get("course_number", "")
                credit = subject.get("credit", 0)
                grade = subject.get("grade", "")
                category = subject.get("category", "")
                
                details.append(f"  - {name} ({course_number}) [{category}] {credit}학점, {grade}")
        
        return "\n".join(details)
    
    def create_basic_transcript_summary(self, transcript_data: Dict) -> str:
        """기본 성적표 요약만 생성 (학기별 상세 정보 제외)"""
        # 학생 기본 정보와 전체 요약만 포함
        basic_info = self.summarizer.extract_student_info(transcript_data)
        summary_info = self.summarizer.extract_summary_info(transcript_data)
        
        return f"{basic_info}\n\n{summary_info}"
    
    def create_full_context(self, user_question: str, transcript_data: Optional[Dict] = None,
                          rag_search_func=None, analysis_result: Optional[Dict] = None,
                          include_semester_details: bool = True) -> str:
        """전체 컨텍스트 생성 (RAG + 성적표 분석 통합)"""
        
        if transcript_data is None:
            return self.create_basic_system_prompt()
        
        # RAG 검색 수행 (더 많은 문서 검색)
        rag_context = ""
        if rag_search_func:
            try:
                # 검색 키워드 확장
                expanded_queries = self.expand_search_queries(user_question)
                all_docs = []
                
                for query in expanded_queries:
                    docs = rag_search_func(query, k=3, max_doc_length=1200)
                    all_docs.extend(docs)
                
                # 중복 제거 및 관련성 높은 문서 선별
                unique_docs = self.deduplicate_and_rank_docs(all_docs, user_question)
                relevant_docs = unique_docs[:5]  # 상위 5개만 선택
                
                rag_context = self.create_rag_context(relevant_docs)
                print(f"🔍 RAG 검색 결과: {len(expanded_queries)}개 쿼리로 {len(relevant_docs)}개 문서 검색됨")
                
                # 검색된 문서 제목들 출력 (디버깅용)
                for i, doc in enumerate(relevant_docs[:3], 1):
                    title = doc.split('\n')[0] if doc else "제목 없음"
                    print(f"  [{i}] {title[:50]}...")
                    
            except Exception as e:
                print(f"⚠️ RAG 검색 실패: {e}")
        
        # 기본 성적표 요약만 생성 (RAG로 필요한 정보만 가져옴)
        transcript_summary = self.create_basic_transcript_summary(transcript_data)
        
        # 분석 결과 요약
        if analysis_result:
            summary = analysis_result.get("summary", "분석 결과 없음")
            regulation = analysis_result.get("regulation", {})
        else:
            summary = "분석을 수행하지 않았습니다."
            regulation = {}
        
        # 최종 시스템 프롬프트 생성
        final_prompt = self.create_improved_system_prompt(
            transcript_summary=transcript_summary,
            regulation=regulation,
            summary=summary,
            rag_context=rag_context,
            analysis_result=analysis_result
        )
        
        # 프롬프트 길이 정보 출력
        stats = self.get_context_statistics(final_prompt)
        print(f"📊 프롬프트 통계: {stats['character_count']}자, {stats['word_count']}단어, 예상토큰 {stats['estimated_tokens']}")
        
        return final_prompt
    
    def expand_search_queries(self, user_question: str) -> List[str]:
        """사용자 질문을 기반으로 검색 쿼리 확장"""
        queries = [user_question]  # 원본 질문
        
        # 학기 관련 질문 처리
        if "학기" in user_question:
            # 연도와 학기 추출
            import re
            year_match = re.search(r'(\d{4})년?도?', user_question)
            semester_match = re.search(r'(\d)학기', user_question)
            
            if year_match and semester_match:
                year = year_match.group(1)
                semester = semester_match.group(1)
                queries.extend([
                    f"{year}년도 {semester}학기",
                    f"{year}년 {semester}학기",
                    f"{year}-{semester}",
                ])
        
        # 과목 관련 질문 처리
        if any(keyword in user_question for keyword in ["과목", "수강", "들었", "이수"]):
            queries.extend([
                "수강 정보",
                "과목 정보",
                "학기별 수강"
            ])
        
        # 성적 관련 질문 처리
        if any(keyword in user_question for keyword in ["성적", "평점", "학점", "A+", "F"]):
            queries.extend([
                "성적 패턴",
                "평점 추이",
                "성적 분포"
            ])
        
        # 졸업 관련 질문 처리
        if any(keyword in user_question for keyword in ["졸업", "요건", "충족"]):
            queries.extend([
                "졸업 요건",
                "졸업 분석",
                "영어 인증"
            ])
        
        return list(set(queries))  # 중복 제거
    
    def deduplicate_and_rank_docs(self, docs: List[str], user_question: str) -> List[str]:
        """문서 중복 제거 및 관련성 기반 순위 매기기"""
        if not docs:
            return []
        
        # 중복 제거 (첫 줄 기준)
        seen_titles = set()
        unique_docs = []
        
        for doc in docs:
            title = doc.split('\n')[0] if doc else ""
            if title not in seen_titles:
                seen_titles.add(title)
                unique_docs.append(doc)
        
        # 관련성 기반 순위 매기기 (간단한 키워드 매칭)
        def calculate_relevance(doc: str) -> int:
            score = 0
            doc_lower = doc.lower()
            question_lower = user_question.lower()
            
            # 직접 키워드 매칭
            for word in question_lower.split():
                if len(word) > 1 and word in doc_lower:
                    score += 1
            
            # 특별 키워드 가중치
            if "학기" in question_lower and "학기" in doc_lower:
                score += 3
            if "과목" in question_lower and any(keyword in doc_lower for keyword in ["과목", "수강"]):
                score += 2
            if "성적" in question_lower and any(keyword in doc_lower for keyword in ["성적", "평점"]):
                score += 2
                
            return score
        
        # 관련성 점수로 정렬
        ranked_docs = sorted(unique_docs, key=calculate_relevance, reverse=True)
        return ranked_docs
    
    def create_layered_context(self, user_question: str, context_layers: Dict[str, str]) -> str:
        """계층별 컨텍스트 구성 (향후 확장용)"""
        # TODO: 계층별 컨텍스트 구성 로직 구현
        # 예: 학생정보 계층, 학사정보 계층, 졸업요건 계층 등
        pass
    
    def optimize_context_length(self, context: str, max_tokens: int = 28000) -> str:
        """컨텍스트 길이 최적화"""
        # 간단한 길이 기반 최적화 (향후 토큰 기반으로 개선 가능)
        estimated_tokens = len(context.split()) * 1.3  # 대략적인 토큰 추정
        
        if estimated_tokens > max_tokens:
            # 컨텍스트를 줄여야 하는 경우
            target_length = int(len(context) * (max_tokens / estimated_tokens))
            context = context[:target_length] + "\n\n[컨텍스트가 길이 제한으로 인해 축약되었습니다]"
            print(f"⚠️ 컨텍스트가 {target_length} 문자로 축약되었습니다.")
        
        return context
    
    def get_context_statistics(self, context: str) -> Dict[str, Any]:
        """컨텍스트 통계 정보"""
        lines = context.split('\n')
        words = context.split()
        
        return {
            "character_count": len(context),
            "word_count": len(words),
            "line_count": len(lines),
            "estimated_tokens": int(len(words) * 1.3),
            "sections": len([line for line in lines if line.startswith("===")])
        }


class HierarchicalContextBuilder(ContextBuilder):
    """계층적 컨텍스트 구성 클래스 (향후 구현)"""
    
    def __init__(self):
        super().__init__()
        self.context_layers = {
            "student": {"priority": 1, "max_tokens": 500},
            "academic": {"priority": 2, "max_tokens": 1000},
            "graduation": {"priority": 3, "max_tokens": 800},
            "rag": {"priority": 4, "max_tokens": 1200}
        }
    
    def create_layered_context(self, user_question: str, context_layers: Dict[str, str]) -> str:
        """계층별 우선순위에 따른 컨텍스트 구성"""
        # TODO: 구체적인 계층별 컨텍스트 구성 로직 구현
        # 우선순위에 따라 토큰 할당 및 컨텍스트 구성
        pass


# 팩토리 함수
def create_context_builder() -> ContextBuilder:
    """컨텍스트 빌더 생성"""
    return ContextBuilder()


def create_hierarchical_context_builder() -> HierarchicalContextBuilder:
    """계층적 컨텍스트 빌더 생성"""
    return HierarchicalContextBuilder() 
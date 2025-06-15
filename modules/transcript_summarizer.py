#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
성적표 JSON 요약 및 RAG 문서 변환 모듈
토큰 사용량을 줄이기 위해 주요 정보만 추출하여 구조화된 텍스트로 변환
"""

import json
from typing import Dict, List, Any, Tuple

class TranscriptSummarizer:
    """성적표 JSON을 요약하여 RAG용 문서로 변환하는 클래스"""
    
    def __init__(self):
        self.grade_to_point = {
            'A+': 4.5, 'A0': 4.0, 'B+': 3.5, 'B0': 3.0, 
            'C+': 2.5, 'C0': 2.0, 'D+': 1.5, 'D0': 1.0, 
            'F': 0.0, 'P': None, 'NP': None
        }
    
    def extract_student_info(self, transcript_data: Dict) -> str:
        """학생 기본 정보 추출 (문자열로 반환)"""
        header = transcript_data.get("ground_truth", {}).get("header", {})
        
        # name 필드 안전하게 처리
        name = header.get('name', '')
        if isinstance(name, list):
            name = name[0] if name else ''
        
        result = [
            "=== 학생 기본 정보 ===",
            f"학번: {header.get('id', '')}",
            f"이름: {name}",
            f"전공: {header.get('major', '')}"
        ]
        
        return "\n".join(result)
    
    def extract_summary_info(self, transcript_data: Dict) -> str:
        """전체 요약 정보 추출"""
        summary = transcript_data.get("ground_truth", {}).get("summary", {})
        totals = summary.get("totals", {})
        categories = summary.get("categories", {})
        
        result = [
            "=== 전체 요약 정보 ===",
            f"총 신청학점: {totals.get('신청학점(A)', 0)}",
            f"총 취득학점: {totals.get('취득학점(B)', 0)}",
            f"전체 평점평균: {totals.get('평점평균(C/B)', 0.0)}",
            "",
            "=== 카테고리별 이수학점 ===",
            f"교양필수: {categories.get('교양필수', 0)}학점",
            f"교양선택: {categories.get('교양선택', 0)}학점",
            f"전공필수: {categories.get('전공필수', 0)}학점",
            f"전공선택: {categories.get('전공선택', 0)}학점"
        ]
        
        return "\n".join(result)
    
    def extract_semester_summary(self, transcript_data: Dict) -> str:
        """학기별 요약 정보 추출"""
        semesters = transcript_data.get("ground_truth", {}).get("semesters", [])
        result = ["=== 학기별 요약 ===\n"]
        
        for semester in semesters:
            year_term = semester.get("year_term", "")
            earned = semester.get("earned", 0)
            gpa = semester.get("gpa", 0.0)
            subjects = semester.get("subjects", [])
            
            # 학기 기본 정보
            result.append(f"{year_term}:")
            result.append(f"  - 신청학점: {earned}학점")
            result.append(f"  - 취득학점: {earned}학점  ")
            result.append(f"  - 학기 평점평균: {gpa}")
            result.append(f"  - 수강과목수: {len(subjects)}과목")
            
            # 학기별 과목 정보 추가
            result.append("\n  수강 과목:")
            for subject in subjects:
                name = subject.get("name", "")
                course_number = subject.get("course_number", "")
                credit = subject.get("credit", 0)
                grade = subject.get("grade", "")
                category = subject.get("category", "")
                
                result.append(f"    - {name} ({course_number})")
                result.append(f"      학점: {credit}학점, 성적: {grade}, 구분: {category}")
            
            result.append("")  # 학기 구분을 위한 빈 줄
        
        return "\n".join(result)
    
    def extract_course_info(self, transcript_data: Dict) -> str:
        """과목별 정보 추출"""
        semesters = transcript_data.get("ground_truth", {}).get("semesters", [])
        all_subjects = []
        
        for semester in semesters:
            all_subjects.extend(semester.get("subjects", []))
        
        # 전공 과목 분류 (전필, 전선, 전공필수, 전공선택 모두 포함)
        major_subjects = [s for s in all_subjects if s.get("category") in ["전필", "전선", "전공필수", "전공선택"]]
        major_subjects.sort(key=lambda x: x.get("course_number", ""))
        
        # 성적 주의 과목 (C+ 이하)
        low_grade_subjects = [s for s in all_subjects if self._is_low_grade(s.get("grade", ""))]
        low_grade_subjects.sort(key=lambda x: x.get("course_number", ""))
        
        # 우수 성적 과목 (A 이상)
        high_grade_subjects = [s for s in all_subjects if self._is_high_grade(s.get("grade", ""))]
        high_grade_subjects.sort(key=lambda x: x.get("course_number", ""))
        
        result = [
            "=== 전공 과목 현황 ===",
            f"전공과목 ({len(major_subjects)}과목):"
        ]
        
        for subject in major_subjects:
            result.append(f"  - {subject.get('name', '')} ({subject.get('course_number', '')}) ({subject.get('credit', 0)}학점, {subject.get('grade', '')})")
        
        result.extend([
            "",
            "=== 성적 주의 과목 (C+ 이하) ==="
        ])
        
        for subject in low_grade_subjects:
            result.append(f"  - {subject.get('name', '')} ({subject.get('course_number', '')}) ({subject.get('credit', 0)}학점, {subject.get('grade', '')})")
        
        result.extend([
            "",
            "=== 우수 성적 과목 (A 이상) ==="
        ])
        
        for subject in high_grade_subjects:
            result.append(f"  - {subject.get('name', '')} ({subject.get('course_number', '')}) ({subject.get('credit', 0)}학점, {subject.get('grade', '')})")
        
        return "\n".join(result)
    
    def extract_graduation_analysis(self, transcript_data: Dict) -> str:
        """졸업요건 기본 정보 추출"""
        summary = transcript_data.get("ground_truth", {}).get("summary", {})
        categories = summary.get("categories", {})
        totals = summary.get("totals", {})
        
        result = [
            "=== 졸업요건 기본 정보 ===",
            f"전공학점: {categories.get('전공필수', 0) + categories.get('전공선택', 0)}학점",
            f"총 취득학점: {totals.get('취득학점(B)', 0)}학점",
            f"전체 평점평균: {totals.get('평점평균(C/B)', 0.0)}"
        ]
        
        return "\n".join(result)
    
    def create_rag_document(self, transcript_data: Dict) -> str:
        """전체 성적표를 RAG용 문서로 변환 (기본 정보만 포함)"""
        # 핵심 정보만 포함 - 모든 메서드가 문자열을 반환하도록 수정됨
        basic_info = self.extract_student_info(transcript_data)
        summary_info = self.extract_summary_info(transcript_data)
        semester_info = self.extract_semester_summary(transcript_data)
        graduation_info = self.extract_graduation_analysis(transcript_data)
        
        sections = [
            basic_info,
            summary_info,
            semester_info,
            graduation_info
        ]
        
        return "\n\n".join(sections)
    
    def create_multiple_rag_docs(self, transcript_data: Dict) -> List[Dict[str, str]]:
        """여러 개의 RAG 문서 생성 (학기별 개별 문서 포함)"""
        docs = []
        
        # 학생 기본 정보
        docs.append({
            "title": "학생 기본 정보",
            "type": "student_info",
            "content": self.extract_student_info(transcript_data)
        })
        
        # 전체 요약 정보
        docs.append({
            "title": "전체 요약 정보",
            "type": "summary",
            "content": self.extract_summary_info(transcript_data)
        })
        
        # 학기별 개별 문서 생성
        semesters = transcript_data.get("ground_truth", {}).get("semesters", [])
        for semester in semesters:
            semester_doc = self.create_individual_semester_doc(semester)
            docs.append(semester_doc)
        
        # 과목별 정보
        docs.append({
            "title": "과목별 정보",
            "type": "course_info", 
            "content": self.extract_course_info(transcript_data)
        })
        
        # 성적 패턴 분석
        docs.append({
            "title": "성적 패턴 분석",
            "type": "grade_analysis",
            "content": self.extract_grade_patterns(transcript_data)
        })
        
        # 졸업요건 기본 정보
        docs.append({
            "title": "졸업요건 기본 정보", 
            "type": "graduation_info",
            "content": self.extract_graduation_analysis(transcript_data)
        })
        
        return docs
    
    def _is_low_grade(self, grade: str) -> bool:
        """낮은 성적 여부 확인"""
        low_grades = ["C+", "C0", "C-", "D+", "D0", "D-", "F"]
        return grade in low_grades
    
    def _is_high_grade(self, grade: str) -> bool:
        """우수 성적 여부 확인"""
        high_grades = ["A+", "A0", "A-"]
        return grade in high_grades
    
    def create_individual_semester_doc(self, semester: Dict) -> Dict[str, str]:
        """개별 학기 문서 생성"""
        year_term = semester.get("year_term", "")
        subjects = semester.get("subjects", [])
        earned = semester.get("earned", 0)
        gpa = semester.get("gpa", 0.0)
        
        content = [
            f"=== {year_term} 수강 정보 ===",
            f"학기 평점: {gpa}",
            f"취득학점: {earned}학점",
            f"수강과목 수: {len(subjects)}과목",
            "",
            "수강 과목:"
        ]
        
        for subject in subjects:
            name = subject.get("name", "")
            course_number = subject.get("course_number", "")
            credit = subject.get("credit", 0)
            grade = subject.get("grade", "")
            category = subject.get("category", "")
            
            content.append(f"  - {name} ({course_number})")
            content.append(f"    학점: {credit}학점, 성적: {grade}, 구분: {category}")
        
        return {
            "title": f"{year_term} 수강 정보",
            "type": "individual_semester",
            "content": "\n".join(content)
        }
    
    def extract_grade_patterns(self, transcript_data: Dict) -> str:
        """성적 패턴 분석"""
        semesters = transcript_data.get("ground_truth", {}).get("semesters", [])
        
        # 학기별 평점 추이
        gpa_trend = []
        for semester in semesters:
            year_term = semester.get("year_term", "")
            gpa = semester.get("gpa", 0.0)
            gpa_trend.append(f"{year_term}: {gpa}")
        
        # 최고/최저 학기
        if semesters:
            best_semester = max(semesters, key=lambda x: x.get("gpa", 0))
            worst_semester = min(semesters, key=lambda x: x.get("gpa", 0))
            
            best_info = f"{best_semester.get('year_term', '')}: {best_semester.get('gpa', 0)}"
            worst_info = f"{worst_semester.get('year_term', '')}: {worst_semester.get('gpa', 0)}"
        else:
            best_info = "정보 없음"
            worst_info = "정보 없음"
        
        # 전체 과목에서 성적 분포
        all_subjects = []
        for semester in semesters:
            all_subjects.extend(semester.get("subjects", []))
        
        grade_counts = {}
        for subject in all_subjects:
            grade = subject.get("grade", "")
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
        grade_distribution = []
        for grade, count in sorted(grade_counts.items()):
            grade_distribution.append(f"{grade}: {count}과목")
        
        content = [
            "=== 성적 패턴 분석 ===",
            "",
            "학기별 평점 추이:",
            "\n".join([f"  {trend}" for trend in gpa_trend]),
            "",
            f"최고 성적 학기: {best_info}",
            f"최저 성적 학기: {worst_info}",
            "",
            "성적 분포:",
            "\n".join([f"  {dist}" for dist in grade_distribution])
        ]
        
        return "\n".join(content)
    
    def get_specific_info(self, transcript_data: Dict, info_type: str) -> str:
        """특정 정보만 추출"""
        if info_type == "student_info":
            return self.extract_student_info(transcript_data)
        elif info_type == "summary":
            return self.extract_summary_info(transcript_data)
        elif info_type == "semesters":
            return self.extract_semester_summary(transcript_data)
        elif info_type == "courses":
            return self.extract_course_info(transcript_data)
        elif info_type == "graduation":
            return self.extract_graduation_analysis(transcript_data)
        else:
            return self.create_rag_document(transcript_data)


def demo_usage():
    """사용 예시"""
    # 성적표 JSON 로드
    transcript_file = r"C:\Users\user\Desktop\DeepLearning\LLM\Project_AI\outputs\transcripts_100\12190002_b6d48c.json"
    
    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    # 요약기 생성
    summarizer = TranscriptSummarizer()
    
    # 전체 RAG 문서 생성
    full_doc = summarizer.create_rag_document(transcript_data)
    print("=== 전체 RAG 문서 ===")
    print(full_doc)
    print(f"\n토큰 수 (대략): {len(full_doc.split())}")
    
    # 섹션별 문서 생성
    print("\n=== 섹션별 문서 ===")
    docs = summarizer.create_multiple_rag_docs(transcript_data)
    for doc in docs:
        print(f"\n[{doc['title']}]")
        print(doc['content'][:2000] + "..." if len(doc['content']) > 200 else doc['content'])
    
    # 특정 정보만 추출
    print("\n=== 졸업요건 분석만 ===")
    graduation_info = summarizer.get_specific_info(transcript_data, "graduation")
    print(graduation_info)


if __name__ == "__main__":
    demo_usage() 
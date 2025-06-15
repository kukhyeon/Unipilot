#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
학사 분석 모듈
성적표 분석 및 졸업요건 검증 등 비즈니스 로직을 담당
졸업요건 확인에 필요한 핵심 정보만 추출하여 분석
"""

import json
from typing import Dict, List, Any, Tuple
from json_converter_text import department_regulations


class AcademicAnalyzer:
    """학사 분석 클래스"""
    
    def __init__(self):
        self.department_regulations = department_regulations
    
    def extract_completed_courses(self, transcript_data: Dict) -> List[str]:
        """이수 완료 과목 추출"""
        completed_courses = []
        
        for semester in transcript_data.get("ground_truth", {}).get("semesters", []):
            for subject in semester.get("subjects", []):
                course_number = subject.get("course_number")
                if course_number:
                    completed_courses.append(course_number)
        
        return completed_courses
    
    def get_regulation_by_major_and_year(self, major: str, year: str) -> Dict:
        """전공과 학년도에 따른 졸업요건 반환"""
        key = f"{major}_{year}"
        result = {}

        # Blossom 기반으로 처리할 학과+학번 키
        blossom_keys = {
            "전기공학과_2024",
            "전자공학과_2024", 
            "정보통신공학과_2024",
            "전기전자공학부_2025",
        }

        if key in blossom_keys:
            blossom_key = f"{year}_{major}"
            if blossom_key in self.department_regulations:
                result.update(self.department_regulations[blossom_key])
        else:
            # RAG용 파일 불러오기 (향후 확장)
            rag_path = f"./rag_dataset/졸업요건_{major}_{year}.json"
            import os
            if os.path.exists(rag_path):
                with open(rag_path, "r", encoding="utf-8") as f:
                    result.update(json.load(f))

        # 공통 졸업요건 병합
        common_path = "./rag_dataset/졸업요건_공통.json"
        import os
        if os.path.exists(common_path):
            with open(common_path, "r", encoding="utf-8") as f:
                common = json.load(f)
                result.update(common)

        if not result:
            raise ValueError(f"졸업요건 데이터를 찾을 수 없음: {major}, {year}")

        return result
    
    def analyze_graduation_requirements(self, transcript_data: Dict) -> Dict[str, Any]:
        """졸업요건 분석 결과 생성"""
        try:
            # 학생 정보 추출
            student_info = transcript_data["ground_truth"]["header"]
            department = student_info["major"].split()[0]
            year = student_info["id"][:4]  # 학번 앞 4자리가 입학년도
            
            # 졸업요건 로딩 (안전한 접근)
            try:
                regulation = self.get_regulation_by_major_and_year(department, year)
            except:
                # 기본 졸업요건 설정
                regulation = {
                    "graduation_credits": 130,
                    "major_required_credits": 65,
                    "required_courses": [],
                    "graduation_project_required": True,
                    "english_certification_required": True,
                    "minimum_gpa": 2.0
                }
            
            # 이수 과목 추출
            completed_courses = self.extract_completed_courses(transcript_data)
            
            # 학점 요건 분석
            summary = transcript_data["ground_truth"]["summary"]
            total_credits = summary["totals"]["취득학점(B)"]
            major_credits = summary["categories"]["전공필수"] + summary["categories"]["전공선택"]
            
            # 필수 과목 이수 여부 확인 (안전한 접근)
            required_courses_rules = regulation.get("required_courses", [])
            remaining_required_courses = self.evaluate_required_courses(
                required_courses_rules, completed_courses
            )
            
            # 종합설계 과목 확인
            capstone_completed = False
            for semester in transcript_data["ground_truth"]["semesters"]:
                for subject in semester["subjects"]:
                    if "종합설계" in subject["name"]:
                        capstone_completed = True
                        break
            
            # 영어 인증 확인
            english_certified = transcript_data["ground_truth"].get("english_certification", "").strip() != ""
            
            # 평점 확인
            gpa = summary["totals"]["평점평균(C/B)"]
            
            # 졸업요건 충족 여부 분석
            requirements_status = {
                "total_credits": {
                    "required": regulation.get("graduation_credits", 130),
                    "completed": total_credits,
                    "satisfied": total_credits >= regulation.get("graduation_credits", 130)
                },
                "major_credits": {
                    "required": regulation.get("major_required_credits", 65),
                    "completed": major_credits,
                    "satisfied": major_credits >= regulation.get("major_required_credits", 65)
                },
                "required_courses": {
                    "remaining": remaining_required_courses,
                    "satisfied": len(remaining_required_courses) == 0
                },
                "capstone": {
                    "required": regulation.get("graduation_project_required", True),
                    "completed": capstone_completed,
                    "satisfied": not regulation.get("graduation_project_required", True) or capstone_completed
                },
                "english_certification": {
                    "required": regulation.get("english_certification_required", True),
                    "completed": english_certified,
                    "satisfied": not regulation.get("english_certification_required", True) or english_certified
                },
                "gpa": {
                    "required": regulation.get("minimum_gpa", 2.0),
                    "current": gpa,
                    "satisfied": gpa >= regulation.get("minimum_gpa", 2.0)
                }
            }
            
            # 전체 졸업요건 충족 여부
            all_requirements_satisfied = all(
                status["satisfied"] for status in requirements_status.values()
            )
            
            # RAG용 요약 정보 생성
            rag_summary = {
                "student_info": student_info,
                "department": department,
                "graduation_year": year,
                "requirements_status": requirements_status,
                "is_graduation_ready": all_requirements_satisfied,
                "remaining_requirements": [
                    f"{key}: {status.get('required', 'N/A')} 필요, {status.get('completed', 'N/A')} 이수"
                    for key, status in requirements_status.items()
                    if not status.get("satisfied", False)
                ]
            }
            
            return rag_summary
            
        except Exception as e:
            return {
                "error": f"졸업요건 분석 중 오류 발생: {str(e)}",
                "is_graduation_ready": False
            }
    
    def evaluate_required_courses(self, required_rules: List[Dict], completed_courses: List[str]) -> List[str]:
        """필수 과목 이수 평가"""
        not_completed = []

        for rule in required_rules:
            rule_type = rule["type"]

            if rule_type == "all":
                for course in rule["courses"]:
                    if course not in completed_courses:
                        not_completed.append(course)

            elif rule_type == "or":
                if not any(course in completed_courses for course in rule["courses"]):
                    not_completed.append(f"{'/'.join(rule['courses'])} 중 1과목 이상")

            elif rule_type == "credit_sum":
                total = sum(credit for course, credit in rule["courses"].items() if course in completed_courses)
                if total < rule["min_credits"]:
                    not_completed.append(f"{'/'.join(rule['courses'].keys())} 중 {rule['min_credits']}학점 이상")

            elif rule_type == "at_least_n":
                count = sum(1 for course in rule["courses"] if course in completed_courses)
                if count < rule["min_count"]:
                    not_completed.append(f"{'/'.join(rule['courses'])} 중 {rule['min_count']}과목 이상")

        return not_completed


# 팩토리 함수
def create_academic_analyzer() -> AcademicAnalyzer:
    """학사 분석기 생성"""
    return AcademicAnalyzer() 
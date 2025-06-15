import os
import json
import pprint
import tempfile
import zipfile
import random
from tqdm import tqdm
from torch.utils.data import Dataset

# 내규 데이터베이스 (첫 번째 코드의 상세한 버전 사용)
department_regulations = {
    "전기전자공학부": {
        "graduation_credits": 130,
        "major_required_credits": 65,
        "required_courses": [
            {
                "type": "or",
                "courses": ["GEB1107", "GEB1108", "GEB1109"]
            },
            {
                "type": "credit_sum",
                "courses": {
                    "EEC2200": 3, "EEC2202": 3, "EEC2204": 3, "EEC2206": 3, "EEC2208": 3,
                    "EEC3200": 1, "EEC3202": 3, "EEC3204": 3, "EEC3206": 4,
                    "EEC3208": 3, "EEC3210": 3
                },
                "min_credits": 12
            },
            {
                "type": "all",
                "courses": [
                    "GEB1117", "GEB1126", "GEB1143", "GEB1151",
                    "GEB1112", "GEB1114",
                    "EEC1100", "EEC2100", "EEC2101", "EEC2102", "EEC2104", "EEC2106", "EEC2108", "EEC4100",
                    "ACE2901", "ACE2902",
                    "CHM1923", "CHM1927",
                    "MTH1901", "MTH1902",
                    "PHY1901", "PHY1902", "PHY1903", "PHY1904",
                    "EEC1102", "EEC1104", "EEC2110"
                ]
            }
        ],
        "graduation_project_required": True,
        "english_certification_required": True,
        "default_semesters": 8,
        "special_requirements": "종합설계 과목 필수 이수, 영어인증 필요"
    },

    "정보통신공학과": {
        "graduation_credits": 130,
        "major_required_credits": 65,
        "required_courses": [
            {
                "type": "or",
                "courses": ["GEB1107", "GEB1108", "GEB1109"]
            },
            {
                "type": "all",
                "courses": [
                    "GEB1116", "GEB1126", "GEB1143", "GEB1151",
                    "GEB1112", "GEB1114",
                    "ACE1204", "ACE2101", "ACE2102", "ACE2105",
                    "ICE1004", "ICE1005",
                    "MTH1001", "MTH1002",
                    "PHY1001", "PHY1002", "PHY1003", "PHY1004",
                    "ICE1001", "ICE1002",
                    "ICE2002", "ICE2003", "ICE2004", "ICE2005", "ICE2006", "ICE2007", "ICE2014",
                    "ICE3001", "ICE4024"
                ]
            }
        ],
        "graduation_project_required": True,
        "english_certification_required": True,
        "default_semesters": 8,
        "special_requirements": "종합설계 과목 필수 이수, 영어인증 필요"
    },

    "전자공학과": {
        "graduation_credits": 130,
        "major_required_credits": 65,
        "required_courses": [
            {
                "type": "or",
                "courses": ["GEB1107", "GEB1108", "GEB1109"]
            },
            {
                "type": "all",
                "courses": [
                    "GEB1116", "GEB1126", "GEB1143", "GEB1151",
                    "GEB1112", "GEB1114",
                    "ACE1302", "ACE2101", "ACE2102", "ACE2105",
                    "CHM1023", "CHM1027",
                    "MTH1001", "MTH1002",
                    "PHY1001", "PHY1002", "PHY1003", "PHY1004",
                    "ECE1211", "ECE2222", "ECE2224", "ECE2240", "ECE2243", "ECE2248",
                    "ECE2250", "ECE2260", "ECE2266", "ECE3320"
                ]
            }
        ],
        "graduation_project_required": True,
        "english_certification_required": True,
        "default_semesters": 8,
        "special_requirements": "종합설계 과목 필수 이수, 영어인증 필요"
    },

    "전기공학과": {
        "graduation_credits": 130,
        "major_required_credits": 65,
        "required_courses": [
            {
                "type": "or",
                "courses": ["GEB1107", "GEB1108", "GEB1109"]
            },
            {
                "type": "or",
                "courses": ["ACE1307", "ACE2103", "ACE2105"]
            },
            {
                "type": "at_least_n",
                "min_count": 3,
                "courses": ["EEE2008", "EEE2201", "EEE3323", "EEE3324", "EEE3114"]
            },
            {
                "type": "all",
                "courses": [
                    "GEB1112", "GEB1114", "GEB1116", "GEB1126", "GEB1143", "GEB1151",
                    "ACE1302", "ACE2101", "ACE2102",
                    "CHM1023", "CHM1027",
                    "MTH1001", "MTH1002",
                    "PHY1001", "PHY1002", "PHY1003", "PHY1004",
                    "EEE1001", "EEE2001", "EEE2002", "EEE2003", "EEE2004", "EEE2005", "EEE2006", "EEE2007",
                    "EEE3001", "EEE3002", "EEE4001"
                ]
            }
        ],
        "graduation_project_required": True,
        "english_certification_required": True,
        "default_semesters": 8,
        "special_requirements": "종합설계 과목 필수 이수, 영어인증 필요"
    },
    
    # 컴퓨터공학과 추가 (두 번째 코드에서)
    "컴퓨터공학과": {
        "graduation_credits": 140,
        "major_required_credits": 50,
        "major_elective_credits": 35,
        "required_courses": [
            {
                "type": "all",
                "courses": ["CSE1001", "CSE2002", "CSE4100"]
            }
        ],
        "graduation_project_required": True,
        "english_certification_required": False,
        "default_semesters": 8,
        "special_requirements": "캡스톤디자인 필수, 인턴십 권장"
    }
}

def evaluate_required_courses(required_rules, completed_courses):
    """첫 번째 코드의 복잡한 내규 평가 함수"""
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

def generate_qa_pairs(transcript_data, regulation):
    """
    성적표와 내규 정보를 바탕으로 다양한 질문-답변 쌍 생성
    첫 번째 코드의 복잡한 내규 체크 로직 적용
    """
    qa_pairs = []
    student_info = transcript_data["ground_truth"]["header"]
    semesters = transcript_data["ground_truth"]["semesters"]
    
    # 전체 이수 학점 계산
    total_credits = sum(semester["earned"] for semester in semesters)
    
    # 과목 코드 추출
    all_courses = []
    for semester in semesters:
        for subject in semester["subjects"]:
            all_courses.append(subject["course_number"])
    
    # 1. 졸업 요건 충족 여부 질문 (첫 번째 코드의 복잡한 로직 사용)
    remaining_credits = regulation["graduation_credits"] - total_credits
    remaining_required_courses = evaluate_required_courses(regulation["required_courses"], all_courses)
    
    graduation_answer = f"현재까지 {total_credits}학점을 이수했으며, 졸업까지 {remaining_credits}학점이 더 필요합니다. "
    if remaining_required_courses:
        graduation_answer += f"아직 이수하지 않은 필수 과목은 {', '.join(remaining_required_courses)}입니다."
    else:
        graduation_answer += "모든 필수 과목을 이수했습니다."
    
    qa_pairs.append((
        f"졸업 요건을 충족했나요?",
        graduation_answer
    ))
    
    # 2. 특정 학기 성적 질문
    for semester in semesters:
        year_term = semester["year_term"]
        gpa = semester["gpa"]
        
        qa_pairs.append((
            f"{year_term}의 평균 학점은 얼마인가요?",
            f"{year_term}의 평균 학점은 {gpa}입니다."
        ))
        
        qa_pairs.append((
            f"{year_term}에 수강한 과목들을 알려주세요.",
            f"{year_term}에 수강한 과목은 {', '.join([subject['name'] for subject in semester['subjects']])}입니다."
        ))
    
    # 3. 전체 평균 학점 질문
    if semesters:
        total_gpa = sum(semester["gpa"] for semester in semesters) / len(semesters)
        qa_pairs.append((
            "전체 평균 학점은 얼마인가요?",
            f"전체 평균 학점은 {total_gpa:.2f}입니다."
        ))
    
    # 4. 특정 과목 성적 질문
    subject_count = 0
    for semester in semesters:
        for subject in semester["subjects"]:
            subject = random.choice(semester["subjects"])
            if subject_count < 3:
                qa_pairs.append((
                    f"{subject['name']} 과목의 성적은 어떻게 되나요?",
                    f"{subject['name']} 과목은 {semester['year_term']}에 {subject['grade']} 학점을 받았습니다."
                ))
                subject_count += 1
    
    # 5. 재수강 과목 확인
    course_count = {}
    for semester in semesters:
        for subject in semester["subjects"]:
            course_number = subject["course_number"]
            if course_number in course_count:
                course_count[course_number] += 1
            else:
                course_count[course_number] = 1
    
    retaken_courses = [k for k, v in course_count.items() if v > 1]
    if retaken_courses:
        retaken_course_names = []
        for course_code in retaken_courses:
            for semester in semesters:
                for subject in semester["subjects"]:
                    if subject["course_number"] == course_code:
                        retaken_course_names.append(subject["name"])
                        break
                if retaken_course_names:
                    break
        
        qa_pairs.append((
            "재수강한 과목이 있나요?",
            f"네, {', '.join(set(retaken_course_names))} 과목을 재수강했습니다."
        ))
    else:
        qa_pairs.append((
            "재수강한 과목이 있나요?",
            "아니요, 재수강한 과목이 없습니다."
        ))

    # 6. 학생 기본 정보 확인
    qa_pairs.append((
        "이 학생의 이름은 무엇인가요?",
        f"이 학생의 이름은 {student_info['name']}입니다."
    ))

    qa_pairs.append((
        "이 성적표의 학번은 어떻게 되나요?",
        f"학번은 {student_info['id']}입니다."
    ))

    qa_pairs.append((
        "전공이 무엇인가요?",
        f"전공은 {student_info['major']}입니다."
    ))

    # 7. 특정 과목명이 정확히 들어갔는지 확인
    for semester in semesters:
        if semester["subjects"]:
            subject = random.choice(semester["subjects"])
            qa_pairs.append((
                f"'{subject['course_number']}' 과목의 정식 명칭은 무엇인가요?",
                f"{subject['course_number']} 과목의 명칭은 '{subject['name']}'입니다."
            ))

    # 8. 과목명과 과목코드 출력
    for semester in semesters:
        for subject in semester["subjects"]:
            subject = random.choice(semester["subjects"])
            qa_pairs.append((
                f"'{subject['name']}' 과목의 과목 코드는 무엇인가요?",
                f"{subject['name']} 과목의 과목 코드는 {subject['course_number']}입니다."
            ))
            break

    # 9. 수강 학기 판별
    for semester in semesters:
        for subject in semester["subjects"]:
            subject = random.choice(semester["subjects"])
            qa_pairs.append((
                f"{subject['name']} 과목은 언제 수강했나요?",
                f"{subject['name']} 과목은 {semester['year_term']}에 수강했습니다."
            ))
            break
    
    # 10. 특별 요구사항 관련 질문 추가
    if regulation.get("special_requirements"):
        qa_pairs.append((
            "졸업을 위한 특별 요구사항이 있나요?",
            f"네, {regulation['special_requirements']}가 필요합니다."
        ))
    
    return qa_pairs

def format_chat_messages_for_llama32(student_info, regulation, question, answer=None):
    """
    Llama 3.2 모델 입력을 위한 메시지 형식 생성
    """
    # 시스템 지시문 구성
    system_instructions = (
        f"당신은 대학생 성적표 분석 도우미입니다. "
        f"학생의 성적표와 학과 내규를 분석하여 질문에 정확하게 답변해 주세요.\n\n"
        f"학생 정보:\n- 학번: {student_info['id']}\n- 이름: {student_info['name']}\n"
        f"- 전공: {student_info['major']}\n\n학과 내규:\n{regulation}\n\n"
    )
    
    # 메시지 구조 생성 (Llama 3.2 형식)
    messages = [
        {'role': 'system', 'content': system_instructions},
        {'role': 'user', 'content': question}
    ]
    
    # 답변이 제공된 경우 (학습용)
    if answer:
        messages.append({'role': 'assistant', 'content': answer})
    
    return messages

def convert_json_to_llama32_format(json_file_path):
    """
    JSON 파일을 Llama 3.2 모델 학습 데이터로 변환
    """
    print(f"JSON 파일 변환 중: {json_file_path}")
    
    # JSON 파일 로드
    with open(json_file_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    # 학생 정보
    student_info = transcript_data["ground_truth"]["header"]
    major = student_info["major"]
    
    # 해당 학과의 내규 가져오기
    department = major.split()[0]  # 예: "전기전자공학부"
    regulation = department_regulations.get(department, department_regulations["전기전자공학부"])
    
    # 질문-답변 쌍 생성
    qa_pairs = generate_qa_pairs(transcript_data, regulation)
    
    # 학습 데이터 생성
    training_data = []
    for question, answer in qa_pairs:
        # 저장 가능한 형태로 학생 정보와 내규 정보 직렬화
        student_info_serializable = {
            "id": student_info["id"],
            "name": student_info["name"],
            "major": student_info["major"]
        }
        
        regulation_str = json.dumps(regulation, ensure_ascii=False)
        
        # Llama 3.2 형식의 메시지 생성
        messages = format_chat_messages_for_llama32(
            student_info,
            regulation_str,
            question,
            answer
        )
        
        # 각 질문-답변 쌍에 대한 학습 데이터 항목
        training_item = {
            "student_info": student_info_serializable,
            "regulation_str": regulation_str,
            "question": question,
            "answer": answer,
            "raw_transcript": transcript_data,
            "messages": messages  # Llama 3.2 형식의 메시지
        }
        training_data.append(training_item)
    
    return training_data

def process_multiple_jsons_for_llama32(json_dir, output_file_path):
    """
    디렉토리 내의 모든 JSON 파일을 Llama 3.2 학습 형식으로 변환하여 JSONL 파일로 저장
    """
    all_training_data = []
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    print(f"총 {len(json_files)}개의 JSON 파일 처리 중...")
    
    for json_file in tqdm(json_files):
        json_file_path = os.path.join(json_dir, json_file)
        try:
            training_data = convert_json_to_llama32_format(json_file_path)
            all_training_data.extend(training_data)
            print(f"- {json_file}: {len(training_data)}개의 질문-답변 쌍 생성")
        except Exception as e:
            print(f"- {json_file} 처리 중 오류 발생: {e}")
    
    # 출력 디렉토리가 없으면 생성
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 학습 데이터를 JSONL 파일로 저장
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in all_training_data:
            serializable_item = {
                "student_info": item["student_info"],
                "regulation_str": item["regulation_str"],
                "question": item["question"],
                "answer": item["answer"],
                "messages": item["messages"]
            }
            f.write(json.dumps(serializable_item, ensure_ascii=False) + '\n')
    
    print(f"\n총 {len(all_training_data)}개의 질문-답변 쌍이 '{output_file_path}' 파일에 저장되었습니다.")
    
    # 처음 몇 개의 데이터 예시 출력
    print("\n== 변환된 데이터 예시 ==")
    for i, item in enumerate(all_training_data[:2]):
        print(f"\n예시 {i+1}:")
        print(f"학생: {item['student_info']['name']} ({item['student_info']['id']})")
        print(f"질문: {item['question']}")
        print(f"답변: {item['answer']}")
        print("\nLlama 3.2 메시지 형식:")
        print("-" * 80)
        print(json.dumps(item['messages'], indent=2, ensure_ascii=False))
        print("-" * 80)
        if i >= 1:  # 2개만 출력
            break

def process_zip_and_generate_output(input_zip_path, output_zip_path):
    """ZIP 파일 처리 함수 (두 번째 코드에서 가져옴)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        extracted_dir = os.path.join(tmpdir, "extracted")
        os.makedirs(extracted_dir, exist_ok=True)

        # Step 1: ZIP 압축 해제
        with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)

        # Step 2: JSON 파일 처리
        llama_data = []
        for filename in tqdm(os.listdir(extracted_dir)):
            if filename.endswith(".json"):
                file_path = os.path.join(extracted_dir, filename)
                try:
                    llama_data.extend(convert_json_to_llama32_format(file_path))
                except Exception as e:
                    print(f"{filename} 처리 중 오류 발생: {e}")

        # Step 3: JSONL 생성
        jsonl_path = os.path.join(tmpdir, "llama3.2_output.jsonl")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in llama_data:
                serializable_item = {
                    "student_info": item["student_info"],
                    "regulation_str": item["regulation_str"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "messages": item["messages"]
                }
                f.write(json.dumps(serializable_item, ensure_ascii=False) + '\n')

        # Step 4: 새로운 ZIP 압축
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(jsonl_path, arcname="llama3.2_output.jsonl")
        
        print(f"\n총 {len(llama_data)}개의 질문-답변 쌍이 ZIP 파일로 저장되었습니다.")

# Llama 3.2를 위한 데이터셋 클래스
class Llama32TranscriptDataset(Dataset):
    """
    Llama 3.2 모델 파인튜닝을 위한 데이터셋 클래스
    개선사항:
    - max_length 기본값을 2316으로 변경 (데이터 손실 0%)
    - 토큰 길이 통계 제공
    - truncation 경고 추가
    """
    def __init__(self, jsonl_file, tokenizer, max_length=2316):
        """
        Args:
            jsonl_file (str): JSONL 파일 경로
            tokenizer: Llama 3.2 모델의 토크나이저
            max_length (int): 최대 토큰 길이 (기본값: 2316 - 전체 데이터 보존)
        """
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncated_count = 0  # 잘린 샘플 개수 추적

        # JSONL 파일 로드
        print(f"JSONL 파일 로드 중: {jsonl_file}")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                if line.strip():  # 빈 줄 무시
                    example = json.loads(line)
                    self.examples.append(example)
        
        print(f"로드된 데이터 수: {len(self.examples)}")
        
        # 토큰 길이 분석 (선택적)
        if len(self.examples) <= 1000:  # 1000개 이하면 전체 분석
            self._analyze_token_lengths()
        else:
            print(f"데이터가 많습니다. 샘플 분석을 원하면 analyze_sample_token_lengths() 메소드를 호출하세요.")

    def _analyze_token_lengths(self, sample_size=1000):
        """토큰 길이 분석"""
        print(f"\n토큰 길이 분석 중 (샘플: {min(sample_size, len(self.examples))}개)...")
        
        token_lengths = []
        for i in range(min(sample_size, len(self.examples))):
            example = self.examples[i]
            messages = example['messages'].copy()
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            tokens = self.tokenizer(text, add_special_tokens=False)
            token_lengths.append(len(tokens.input_ids))
        
        import numpy as np
        token_lengths = np.array(token_lengths)
        
        print(f"토큰 길이 통계:")
        print(f"- 평균: {token_lengths.mean():.1f} 토큰")
        print(f"- 최대: {token_lengths.max()} 토큰")
        print(f"- 최소: {token_lengths.min()} 토큰")
        print(f"- 95%ile: {np.percentile(token_lengths, 95):.0f} 토큰")
        
        truncated = np.sum(token_lengths > self.max_length)
        print(f"- max_length={self.max_length} 기준 잘릴 샘플: {truncated}개 ({truncated/len(token_lengths)*100:.1f}%)")
        
        if truncated > 0:
            print(f"⚠️  권장사항: 데이터 손실을 방지하려면 max_length를 {token_lengths.max()}로 설정하세요.")

    def analyze_sample_token_lengths(self, sample_size=1000):
        """외부에서 호출 가능한 토큰 길이 분석"""
        self._analyze_token_lengths(sample_size)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # 메시지 복사 (원본 수정 방지)
        messages = example['messages'].copy()
        
        # 채팅 형식 적용
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 원본 길이 확인 (truncation 경고용)
        original_tokens = self.tokenizer(text, add_special_tokens=False)
        original_length = len(original_tokens.input_ids)
        
        # 토큰화 (truncation 적용)
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # truncation 발생 시 카운트
        if original_length > self.max_length:
            self.truncated_count += 1
            # 첫 번째 truncation 발생 시에만 경고
            if self.truncated_count == 1:
                print(f"⚠️  경고: 토큰이 잘리고 있습니다. (원본: {original_length}, 제한: {self.max_length})")
                print(f"   데이터 손실을 방지하려면 max_length를 늘려주세요.")
        
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": inputs.input_ids.squeeze().clone(),
            "original_length": original_length  # 디버깅용
        }
    
    def get_truncation_stats(self):
        """truncation 통계 반환"""
        return {
            "total_samples": len(self.examples),
            "truncated_samples": self.truncated_count,
            "truncation_rate": self.truncated_count / len(self.examples) * 100 if len(self.examples) > 0 else 0
        }

# 메인 실행
if __name__ == "__main__":
    import sys
    
    # 실행 모드 선택
    if len(sys.argv) > 1 and sys.argv[1] == "zip":
        # ZIP 파일 처리 모드
        input_zip = sys.argv[2] if len(sys.argv) > 2 else "input_json.zip"
        output_zip = sys.argv[3] if len(sys.argv) > 3 else "output_jsonL.zip"
        process_zip_and_generate_output(input_zip, output_zip)
    else:
        # 디렉토리 처리 모드 (기본)
        json_dir = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\user\Desktop\DeepLearning\LLM\Project_AI\outputs\transcript_jsons"
        output_file_path = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\user\Desktop\DeepLearning\LLM\Project_AI\outputs\training_data\llama32_training_data.jsonl"
        
        # Llama 3.2 형식으로 데이터 변환 및 저장
        process_multiple_jsons_for_llama32(json_dir, output_file_path)
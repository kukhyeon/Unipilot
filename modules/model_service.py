#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모델 서비스 모듈
LLM 모델 로딩과 추론만 담당하며 컨텍스트 구성은 외부에서 받음
"""

import os
import torch
import time
import datetime
from typing import Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class ModelService:
    """LLM 모델 서비스 클래스"""
    
    def __init__(self, model_path: str = "C:/Users/user/Desktop/DeepLearning/LLM/Qwen2.5-14B-Instruct"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.model_loaded = False
        
        # 로그 관련
        self.log_dir = "conversation_logs"
        self._setup_log_directory()
        
        # CUDA 최적화 설정
        self._setup_cuda_optimization()
    
    def _setup_cuda_optimization(self) -> None:
        """CUDA 최적화 설정"""
        print("⚡ PyTorch 및 CUDA 최적화 설정 중...")
        
        # 메모리 최적화 환경변수 설정
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 디버깅용
        
        if torch.cuda.is_available():
            # CUDA 최적화 설정
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # 메모리 정리
            torch.cuda.empty_cache()
            print("✅ CUDA 최적화 설정 완료")
            
            print(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
            print(f"📊 CUDA 버전: {torch.version.cuda}")
            print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("❌ CUDA를 사용할 수 없습니다. CPU로 실행됩니다.")
    
    def _setup_log_directory(self) -> None:
        """로그 디렉토리 생성"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"📁 로그 디렉토리 생성: {self.log_dir}")
    
    def load_model(self) -> bool:
        """모델 로딩"""
        try:
            print("🚀 Qwen 2.5 14B 모델을 로딩 중입니다... (24GB VRAM에 최적화)")
            
            # 토크나이저 로드
            print("📝 토크나이저 로딩 중...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 4bit 양자화 설정
            print("📦 Qwen 2.5 14B 모델 로딩 중... (4bit 양자화로 속도 향상)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={0: "20GB", "cpu": "16GB"},
            )
            
            # 모델 로딩 후 메모리 상태 확인
            if self.device == "cuda":
                print(f"✅ 모델이 로드되었습니다!")
                print(f"💾 GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
                print(f"💾 GPU 메모리 예약: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
                torch.cuda.empty_cache()
                print("🧹 GPU 메모리 캐시 정리 완료")
            
            self.model_loaded = True
            print("✅ Qwen 2.5 14B 모델 로딩 완료! (4bit 양자화)")
            return True
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            self.model_loaded = False
            return False
    
    def generate_response(self, messages: list, max_new_tokens: int = 512, 
                         do_sample: bool = False, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """응답 생성"""
        if not self.model_loaded:
            raise RuntimeError("모델이 로딩되지 않았습니다. load_model()을 먼저 호출하세요.")
        
        start_time = time.time()
        initial_gpu_memory = torch.cuda.memory_allocated() / 1024**3 if self.device == "cuda" else 0
        
        # 메시지를 채팅 템플릿에 적용
        chat_template = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 토크나이징
        inputs = self.tokenizer.encode_plus(
            chat_template,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=28000
        )
        
        # GPU로 입력 데이터 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # CUDA 메모리 상태 확인
        if self.device == "cuda":
            print(f"💾 추론 전 GPU 메모리: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        
        # 추론 수행
        print("🤖 Qwen 2.5 14B 추론 중...")
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                # 경고 방지를 위해 sampling 관련 파라미터 명시적으로 None 설정
                temperature=None,
                top_p=None,
                top_k=None,
                **kwargs
            )
        
        # 응답 추출
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        input_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        
        if full_response.startswith(input_text):
            response = full_response[len(input_text):].strip()
        else:
            if "<|im_start|>assistant" in full_response:
                response = full_response.split("<|im_start|>assistant")[-1].strip()
                if response.startswith("\n"):
                    response = response[1:]
            else:
                response = full_response
        
        # 불필요한 토큰 제거
        if response.startswith("\n"):
            response = response[1:]
        if "<|im_end|>" in response:
            response = response.replace("<|im_end|>", "").strip()
        
        print("✅ 답변 생성 완료!")
        
        # 처리 시간 및 메모리 정보 수집
        end_time = time.time()
        processing_time = end_time - start_time
        final_gpu_memory = torch.cuda.memory_allocated() / 1024**3 if self.device == "cuda" else 0
        
        metadata = {
            "processing_time_seconds": round(processing_time, 2),
            "input_tokens": len(self.tokenizer.encode(chat_template)),
            "output_tokens": len(self.tokenizer.encode(response)),
            "gpu_memory_initial_gb": round(initial_gpu_memory, 2),
            "gpu_memory_final_gb": round(final_gpu_memory, 2),
            "model_name": self.model_path,
            "quantization": "4bit",
            "device": self.device
        }
        
        # CUDA 메모리 정리
        if self.device == "cuda":
            print(f"💾 추론 후 GPU 메모리: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            torch.cuda.empty_cache()
            print("🧹 GPU 메모리 캐시 정리 완료")
        
        print(f"⏱️ 총 처리 시간: {processing_time:.2f}초")
        return response, metadata
    
    def save_conversation_log(self, user_question: str, model_response: str, 
                            metadata: Dict[str, Any], system_prompt: str) -> None:
        """대화 로그를 실시간으로 TXT 파일에 저장"""
        today = datetime.datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(self.log_dir, f"qwen_conversation_{today}.txt")
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        memory_usage = f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB" if torch.cuda.is_available() else "N/A"
        
        log_entry = f"""
{'='*80}
대화 기록 #{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}
{'='*80}
⏰ 시간: {timestamp}
🖥️ 장치: {metadata.get('device', 'N/A')}
💾 GPU 메모리: {memory_usage}
⚡ 처리 시간: {metadata.get('processing_time_seconds', 0):.2f}초

📝 시스템 프롬프트:
{system_prompt}

❓ 사용자 질문:
{user_question}

🤖 AI 응답:
{model_response}

📊 통계:
- 질문 길이: {len(user_question)}자
- 응답 길이: {len(model_response)}자
- 입력 토큰: {metadata.get('input_tokens', 'N/A')}개
- 출력 토큰: {metadata.get('output_tokens', 'N/A')}개

"""
        
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
            print(f"💾 로그 저장 완료: {log_file}")
        except Exception as e:
            print(f"❌ 로그 저장 실패: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "model_loaded": self.model_loaded,
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
            "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        }
    
    def cleanup(self) -> None:
        """모델 및 메모리 정리"""
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.model_loaded = False
        print("🗑️ 모델 및 메모리 정리 완료")


# 팩토리 함수
def create_model_service(model_path: str = "C:/Users/user/Desktop/DeepLearning/LLM/Qwen2.5-14B-Instruct") -> ModelService:
    """모델 서비스 생성"""
    return ModelService(model_path) 
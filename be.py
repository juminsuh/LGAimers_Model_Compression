import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from datasets import Dataset

# ==========================================
# 1. 기본 설정
# ==========================================
MODEL_ID = "./base_model"
OUTPUT_DIR = "./submit_be/model"

# ==========================================
# 2. 데이터셋 준비 (사용자 맞춤형)
# ==========================================
# 사용자님께서 준비하신 고품질 데이터셋을 아래 리스트에 채워주세요.
# 텍스트가 다양하고 길수록(최대 2048토큰) 오차 보상이 정교해집니다.
# (최소 128개 이상의 문장을 권장합니다.)
my_custom_data = [
    "여기에 사용자님의 첫 번째 테스트 데이터를 넣으세요.",
    "두 번째 데이터를 넣으세요. 길고 복잡한 문장일수록 좋습니다.",
    # ... 계속 추가 ...
] * 64 # 임시로 개수를 늘리기 위한 곱하기 (실제 데이터 삽입 시 지워주세요)

# llmcompressor가 읽을 수 있도록 Dataset 객체로 변환
dataset = Dataset.from_dict({"text": my_custom_data})

# ==========================================
# 3. 모델 로드
# ==========================================
print("원본 16비트 모델을 불러오는 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    dtype="auto" 
)

# ==========================================
# 4. 양자화 레시피 설정 (사용자 아이디어 집약)
# ==========================================
recipe = [
    GPTQModifier(
        targets="Linear",     # 모든 행렬 곱셈 층 타겟
        scheme="W4A16",       # 가중치는 4비트로 깎고, 연산은 16비트로 수행 (vLLM 최고 효율)
        ignore=["lm_head"],   # 최종 출력층은 16비트 유지 (정확도 보호)
        
        # [핵심 1] 오차 보상 연산 (Hessian 기반)
        sequential_update=True, 
        
        # [핵심 2] 튀는 값을 무시하고 밀집된 곳을 잡기 위한 댐핑 비율
        # (이전에는 에러가 났지만 GPTQModifier에서는 정상 작동하는 고급 옵션입니다)
        dampening_frac=0.01   
    )
]

# ==========================================
# 5. 양자화 실행 (오랜 시간 소요)
# ==========================================
# 각 층마다 16비트와 4비트를 비교하며 오차를 보상하는 연산이 들어갑니다.
# RTX 3060 환경에서 15분 ~ 30분 정도 걸릴 수 있습니다.
print(f"GPTQ 4비트 양자화 시작! (데이터 {len(my_custom_data)}개 기준)")
oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
    num_calibration_samples=len(my_custom_data),
    max_seq_length=1024 # 메모리가 부족하면 512로 줄이세요
)

# ==========================================
# 6. 최종 파일 저장
# ==========================================
print("양자화 완료! 파일을 저장합니다...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"모든 과정이 끝났습니다. 최종 모델이 '{OUTPUT_DIR}'에 저장되었습니다.")
import os
import torch
import shutil
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# 1. 환경 설정
MODEL_ID = "./base_model" # 원본 EXAONE-4.0-1.2B 경로
OUT_DIR  = "./model"      # 최종 저장 경로

# 2. 보정된 캘리브레이션 데이터셋 구축
# 기존 기술 문장 + QA 방어 + IFEval 기반 instruction-following 샘플 통합
calibration_texts = [
    # [그룹 A] 기존 고득점 기술 도메인 문장
    "인공지능 경량화는 모델의 크기를 줄이는 기술입니다.",
    "LG AI Research의 EXAONE 모델은 뛰어난 성능을 보여줍니다.",
    "딥러닝 모델의 파라미터를 줄이면 추론 속도가 빨라집니다.",
    "양자화는 정밀도를 FP16에서 INT8로 낮추는 과정입니다.",
    "해커톤에서 좋은 성과를 거두기 위해 최적화를 수행합니다.",
    "vLLM은 매우 빠른 추론 속도를 제공하는 라이브러리입니다.",
    "데이터셋을 사용하여 모델의 활성화 분포를 분석합니다.",
    "파이썬과 파이토치를 사용하여 모델을 학습시킵니다.",
    "The transformer architecture uses self-attention mechanisms.",
    "Quantization reduces memory and computational costs.",
    "Large language models are changing how we use technology.",
    "Explain the difference between pruning and quantization.",
    
    # [그룹 B] QA 방어 (추론 논리 보존)
    "질문: 사과가 나무에서 떨어지는 현상과 관련된 물리 법칙은?\n답변: 만유인력의 법칙입니다.",
    "질문: 대한민국에서 가장 높은 산의 이름은?\n답변: 제주도 한라산입니다.",
    
    # [그룹 C] IFEval 기반 instruction-following 샘플 (제약 조건 준수 능력 보존)
    "질문: 인공지능에 대해 설명하되, 응답에 쉼표(,)를 절대 사용하지 마세요.\n답변: 인공지능은 인간의 지능을 기계로 구현한 기술이며 현대 사회의 핵심 동력입니다.",
    "질문: 다음 내용을 요약하세요. 단, 반드시 모든 문장을 '~니다' 체로 마무리하세요.\n답변: 이 보고서는 AI의 미래를 다룹니다. 기술 혁신이 가속화될 것으로 보입니다.",
    "질문: 여행 계획을 작성하되, '일본'이라는 단어를 반드시 3번 이상 포함하세요.\n답변: 일본 여행은 즐겁습니다. 일본 도쿄에는 맛집이 많습니다. 일본 온천 방문을 추천합니다.",
    "질문: 아래 정보를 바탕으로 JSON 형식의 응답을 생성하세요.\n답변: {\"이름\": \"EXAONE\", \"버전\": \"4.0\", \"파라미터\": \"1.2B\"}",
    "질문: '사랑'을 주제로 시를 쓰되, 대괄호 [ ]를 사용하여 제목을 강조하세요.\n답변: [사랑의 깊이] 마음속에 피어난 따뜻한 온기가 온 세상을 붉게 물들입니다."
] * 16 # 총 샘플 수 조절

dataset = Dataset.from_dict({"text": calibration_texts})

# 3. 모델 및 토크나이저 로드
print("[INFO] 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 4. W8A8 양자화 레시피 설정 (63.2점 기록 레시피)
recipe = [
    QuantizationModifier(
        targets="Linear",
        scheme="W8A8",       # 가중치와 활성화 모두 8비트 정수 양자화
        ignore=["lm_head"]   # 최종 출력층은 정밀도 유지를 위해 제외
    )
]

# 5. 양자화 실행 (Oneshot)
print(f"[INFO] {len(calibration_texts)}개 샘플로 양자화 시작...")
oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
    max_seq_length=512,      # IFEval 문장의 복잡도를 고려해 512로 설정
    num_calibration_samples=len(calibration_texts)
)

# 6. 모델 저장 및 제출 파일 생성
print(f"[INFO] 모델 저장: {OUT_DIR}")
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("[INFO] 제출 zip 파일 생성 중...")
# submit.zip 내부에 model/ 폴더가 있도록 구성
if os.path.exists("qacal_submit.zip"):
    os.remove("qacal_submit.zip")
shutil.make_archive("qacal_submit", "zip", root_dir=".", base_dir="model")

print(f"[INFO] ✅ 완료! 최종 파일 크기: {os.path.getsize('qacal_submit.zip') / (1024**3):.2f} GB")

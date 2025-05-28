# 꼬마환경탐정 ♻️
어린이를 위한 AI 기반 분리수거 교육 플랫폼입니다.  
Gradio로 프론트 개발 되었습니다.
프론트 화면을 통해 **음성이나 이미지**로 질문을 하면,  
친절한 AI 탐정이 올바른 분리수거 방법을 알려줍니다.

---

## 🧠 주요 기능
- 🎤 음성 인식 → GPT-4o를 통한 분리수거 안내
- 📷 이미지 분류 → Custom Vision으로 재활용 품목 식별
- 🌱 한글 태그 자동 변환 및 설명 제공

---

## 🛠 실행 방법

```bash
# 1. 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 2. 필요한 패키지 설치
pip install -r requirements.txt

# 3. 실행
python main.py

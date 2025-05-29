# 실행방법
# 1. 필요한 라이브러리 설치
# 1-1. 처음에만 : (1) python3 -m venv venv (2) pip install -r requirements.txt
# 2. source venv/bin/activate
# 3. python main.py

# main.py
import gradio as gr
import requests
import openai
import speech_recognition as sr
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os
import tempfile
from gtts import gTTS
from playsound import playsound
import uuid

from dotenv import load_dotenv
load_dotenv()

# Azure OpenAI 설정
client = openai.AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Custom Vision 설정
CUSTOM_VISION_ENDPOINT = os.getenv("CUSTOM_VISION_ENDPOINT")
CUSTOM_VISION_KEY = os.getenv("CUSTOM_VISION_KEY")
CUSTOM_VISION_PROJECT_ID = os.getenv("CUSTOM_VISION_PROJECT_ID")
CUSTOM_VISION_ITERATION_NAME = os.getenv("CUSTOM_VISION_ITERATION_NAME")

# tts 기능
def text_to_speech(text: str):
    tmp_path = f"/tmp/{uuid.uuid4().hex}.mp3"
    tts = gTTS(text, lang='ko')
    tts.save(tmp_path)
    return tmp_path

# 음성 인식 함수
def handle_voice_input(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        user_text = recognizer.recognize_google(audio, language="ko-KR")
    except sr.UnknownValueError:
        return "음성을 인식하지 못했어요. 다시 말씀해 주세요."
    except sr.RequestError:
        return "음성 인식 서비스에 문제가 발생했어요."

    response = client.chat.completions.create(
        model="a24-gpt-4o-mini",
        messages=[
            {"role": "system", "content": "친절한 분리수거 안내 도우미입니다. 어린이들을 대상으로 알려주는 거니까 이모티콘 많이 섞어서 답변해주세요."},
            {"role": "user", "content": user_text}
        ]
    )
    answer = response.choices[0].message.content.strip()
    return f"""
### 🔍 탐정의 대답
<div style="border:1px solid #D8D8DA; border-radius:8px; padding:12px; background-color:#ffffff;">
{answer}
</div>
"""

def classify_and_explain(image):
    image.save("temp.jpg")
    with open("temp.jpg", "rb") as f:
        img_data = f.read()

    headers = {
        "Prediction-Key": CUSTOM_VISION_KEY,
        "Content-Type": "application/octet-stream"
    }
    url = f"{CUSTOM_VISION_ENDPOINT}/customvision/v3.0/Prediction/{CUSTOM_VISION_PROJECT_ID}/classify/iterations/{CUSTOM_VISION_ITERATION_NAME}/image"
    response = requests.post(url, headers=headers, data=img_data)

    try:
        predictions = response.json()["predictions"]
        if not predictions:
            return "이미지를 인식할 수 없어요. 다시 시도해 주세요.", "", None
    except (KeyError, ValueError):
        return "이미지 분석 중 오류가 발생했어요.", "", None

    top_result = predictions[0]["tagName"]

    tag_kor_map = {
        'vinyl': '비닐류',
        'styrofoam': '스티로폼',
        'glass': '유리병',
        'clothes': '의류',
        'paper': '종이류',
        'can': '캔류',
        'computer': '컴퓨터',
        'battery': '폐건전지',
        'fluorescentlamp': '폐형광등',
        'plastic': '플라스틱류'
    }
    top_result_kor = tag_kor_map.get(top_result, top_result)

    prompt = f"'{top_result_kor}'는 어떤 재활용 품목인가요? 어떻게 분리배출해야 하나요? 어린이를 위한 거니까 이모티콘 많이 섞어서, 친절하게 설명해줘."
    completion = client.chat.completions.create(
        model="a24-gpt-4o-mini",
        messages=[
            {"role": "system", "content": "친절한 분리수거 안내 도우미입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    explanation = completion.choices[0].message.content.strip()

    # ✅ TTS mp3 경로 생성
    mp3_path = text_to_speech(explanation)

    # ✅ 텍스트 출력 + mp3 경로 전달
    answer_text = f"""### 🔍 탐정의 대답  
<div style="border:1px solid #D8D8DA; border-radius:8px; padding:12px; background-color:#ffffff;">{top_result_kor}</div>"""

    explanation_text = f"""### ♻️이렇게 버려요!  
<div style="border:1px solid #D8D8DA; border-radius:8px; padding:12px; background-color:#ffffff;">{explanation}<br><br>👍 환경을 생각하는 멋진 선택이에요 🌱</div>"""

    return answer_text, explanation_text, mp3_path

#마크다운
def process_text(text):
    return f"### 결과입니다\n- 입력: **{text}**\n- 처리 완료!"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

#css 스타일을 적용하기 위한 Gradio Blocks
with gr.Blocks(css="""
footer, .svelte-1ipelgc, .wrap.svelte-1ipelgc {
    display: none !important;
}

/* 헤더 - 로고 스타일 */
.logo-title {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 0;
}
.header-bar {
    background-color: #d0f0c0;
    padding: 20px 30px;
    border-radius: 0 0 16px 16px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
}
.logo-block {
    display: flex;
    align-items: center;
    gap: 18px;
}
.header-bar {
    background-color: #d0f0c0;
    padding: 16px 24px;
    border-radius: 0 0 16px 16px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.header-left {
    display: flex;
    align-items: center;
    gap: 16px;
}
.header-logo {
    width: 72px;
    height: auto;
    border-radius: 12px;
}
.brand-text {
    display: flex;
    flex-direction: column;
}
.brand-title {
    font-size: 24px;
    font-weight: bold;
    color: #2e7d32;
}
.brand-sub {
    font-size: 15px;
    color: #4d774e;
    margin-top: 2px;
}
.contact-info {
    font-size: 14px;
    color: #333;
    text-align: right;
    line-height: 1.5;
}
.sub-description {
    font-size: 16px;
    color: #4d774e;
    font-weight: 500;
    text-align: center;
}
.fade-in {
    opacity: 0;
    transform: scale(0.95);
    animation: fadeIn 0.6s ease-out forwards;
}
@keyframes fadeIn {
    to {
        opacity: 1;
        transform: scale(1);
    }
}
               
/* 캐릭터 선택 부분 */
.character-choice-section {
    display: flex;
    justify-content: center;
    align-items: flex-end;
    gap: 28px;
    background-color: #f3f3f3;
    padding: 24px;
    border-radius: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.choice-bubble {
    display: flex;
    align-items: center;
    gap: 12px;
}                    
.character-line {
    display: flex;
    justify-content: center;
    gap: 40px;
    background-color: #f3f3f3;
    padding: 30px;
    border-radius: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.character-img {
    width: 100px;
    height: auto;
    transition: transform 0.3s;
}
.character-img:hover {
    transform: scale(1.1);
}
.character-wrapper {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.character-hover-msg {
    position: absolute;
    top: -40px;
    background: #ffffff;
    color: #333;
    border-radius: 12px;
    padding: 8px 12px;
    font-size: 14px;
    font-weight: bold;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    opacity: 0;
    transform: scale(0.8);
    transition: all 0.3s ease;
    white-space: nowrap;
    z-index: 5;
}
.character-wrapper {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
}
.character-wrapper:hover .character-hover-msg {
    opacity: 1;
    transform: scale(1);
}
/* 말풍선 */
.round-msg {
    position: relative;
    width: 130px;
    height: 130px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    font-weight: bold;
    color: white;
    padding: 10px;
    white-space: pre-line;
    border: none;
    cursor: pointer;
}
.left-msg::after {
    content: "";
    position: absolute;
    bottom: -16px;
    left: 25%; /* 왼쪽으로 치우치게 */
    transform: translateX(-50%);
    border-left: 10px solid transparent;
    border-right: 10px solid transparent;
    border-top: 14px solid #00aaff;
}
.right-msg::after {
    content: "";
    position: absolute;
    bottom: -16px;
    left: 75%; /* 오른쪽으로 치우치게 */
    transform: translateX(-50%);
    border-left: 10px solid transparent;
    border-right: 10px solid transparent;
    border-top: 14px solid #ff3344;
}
.choice-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
}
.left-msg {
    background-color: #00aaff;
    color: white;
}
.speech-bubble {
    position: relative;
    width: 300px;         
    min-height: 90px;    
    padding: 12px 16px;   
    border-radius: 30px;
    display: flex;
    align-items: center; 
    justify-content: center;
    font-size: 16px;
    font-weight: bold;
    color: white;
    white-space: normal;  
    text-align: center;
    border: none;
    cursor: pointer;
    line-height: 1.4;    
}

.bubble-text{
    color: white;
}
.bubble-text:hover {
    transform: scale(1.1);
    transition: transform 0.2s ease;
}
      
.left-bubble {
    background-color: #00aaff !important;
    color: white;
}
.left-bubble::after {
    content: "";
    position: absolute;
    /* left 꼬리 위치 보정 */
    left: -12px;
    border-right: 14px solid #00aaff;    top: 50%;
    transform: translateY(-50%);
    border-top: 10px solid transparent;
    border-bottom: 10px solid transparent;
    border-right: 14px solid #00aaff;
}
.right-bubble {
    background-color: #ff3344;
}
.right-bubble::after {
    content: "";
    position: absolute;
    top: 50%;
    right: -12px;
    transform: translateY(-50%);
    border-top: 10px solid transparent;
    border-bottom: 10px solid transparent;
    border-left: 14px solid #ff3344;
}
               
.right-msg {
    background-color: #ff3344;
    color: white;
}
.character-section {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 40px;
    padding: 20px;
    background-color: #f3f3f3;
    border-radius: 16px;
    margin-bottom: 20px;
    flex-wrap: wrap;
    text-align: center;
    flex-direction: row;
}
#ai-message, #ai-tip {
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 16px;
    margin-top: 10px;
}
.bouncy-icon {
    width: 120px;
    height: auto;
    border-radius: 50%;
    animation: bounce 1s infinite;
}

@keyframes bounce {
    0%, 100% {
        transform: translateY(0px);
    }
    50% {
        transform: translateY(-6px);
    }

}
               
.tool-section {
    background-color: #ffffff;
    border: 2px solid #d0f0c0;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 630px;
}

.tool-section h3 {
    font-size: 20px;
    font-weight: bold;
    color: #2e7d32;
    margin-bottom: 16px;
}

.tool-section .gr-microphone,
.tool-section .gr-image {
    width: 100%;
    max-width: 400px;
}

.tool-section .gr-textbox {
    margin-top: 16px;
    width: 100%;
    max-width: 400px;
}
               
 #answer-box {
        border: 1px solid #D8D8DA;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        white-space: pre-wrap;
        min-height: 100px;
}

""") as demo:

    selected = gr.State(value=False)

    with gr.Column():

        gr.HTML("""
            <div class="header-bar">
            <div class="header-left">
                <img src="/static/logo.png" alt="꼬마환경탐정 로고" class="header-logo" />
                <div class="brand-text">
                <div class="brand-title">꼬마환경탐정</div>
                <div class="brand-sub">지구를 지키는 작지만 큰 실천, 지금 시작해요! 🌱</div>
                </div>
            </div>
            <div class="contact-info">
                <div><strong>문의:</strong> Kideco@kidsmission.org</div>
                <div><strong>운영:</strong> 꼬마환경탐정팀</div>
            </div>
            </div>
        """)

        gr.HTML("""
        <!-- 탐정 인삿말 -->
        <div style="display: flex; align-items: center; justify-content: center; gap: 16px; margin-bottom: 24px;">
            <img src="/static/icon.png" alt="캐릭터" class="bouncy-icon">
            <div>
                <div style="text-align: left; font-size: 22px; font-weight: bold; margin-bottom: 6px;">
                    안녕! 나는 <span style="color: #2e7d32;">꼬마환경탐정</span>이야!
                </div>
                <div style="text-align: left; font-size: 18px;">
                    오늘도 <strong>분리수거 미션</strong>을 함께할게!
                </div>
            </div>
        </div>

        <div class="character-choice-section">
            <div style="width: 100%; text-align: center; font-size: 23px; font-weight: bold; margin-bottom: 5px;">
            둘 중 어떤 친구를 선택할래요? 클릭해보세요!
            </div>
        <div class="choice-bubble">
            <div class="character-wrapper">
                <div class="character-hover-msg">분리수거를 올바르게 하면 지구가 웃어요! 😊</div>
                <img src="/static/good.png" class="character-img" />
            </div>
            <button class="speech-bubble left-bubble" onclick="document.querySelector('#good-button').click()">
                <span class="bubble-text">분리수거 잘하면<br>지구가 깨끗해져요~!</span>
            </button>
        </div>

        <div class="choice-bubble">
            <button class="speech-bubble right-bubble" onclick="document.querySelector('#bad-button').click()">
                <span class="bubble-text">그냥 다 한꺼번에 버려~<br>귀찮잖아!</span>
            </button>
            <div class="character-wrapper">
                <div class="character-hover-msg">정말 나를 선택할 거야...? 지구가 아파요 🥲</div>
                <img src="/static/bad.png" class="character-img" />
            </div>
        </div>
        </div>
        """)

        ai_message = gr.HTML(visible=False, elem_id="ai-message")
        ai_tip = gr.HTML(visible=False, elem_id="ai-tip")

        with gr.Row(visible=False) as tools_row:
            with gr.Column():
                with gr.Column(elem_classes="tool-section"):
                    gr.HTML("<h3>🎤 말로 물어보세요!</h3>")
                    voice_input = gr.Microphone(label="", type="filepath")
                    voice_output = gr.Markdown(label="", elem_id="answer-box")
            with gr.Column():
                with gr.Column(elem_classes="tool-section"):
                    gr.HTML("<h3>📷 사진을 올려보세요!</h3><p>사진을 찍을 때는 하나의 물건만 찍어주세요! \n 📸 여러 개가 있으면 AI가 헷갈릴 수 있어요.</p>")
                    # 이미지 입력
                    image_input = gr.Image(label="", type="pil")

                    #텍스트 출력
                    result = gr.Markdown(label="결과", elem_id="answer-box")
                    howto = gr.Markdown(label="이렇게 버려요!", elem_id="answer-box")

                     # ✅ 음성 재생용 추가
                    play_button = gr.Button("▶️ 음성 재생")
                    audio_output = gr.Audio()

                    # ✅ 내부적으로 음성 경로 저장할 상태
                    tts_path_state = gr.State()

        def good_selected():
            return (
                gr.update(value="<div style='font-size: 26px;'>💡 좋은 생각이야! 탐정에게 물어보자!</div>", visible=True),
                gr.update(visible=True)
            )

        def bad_selected():
            return (
                gr.update(value="<div style='font-size: 26px;'>📚 그러면 안 돼! 분리수거를 같이 배워보자!", visible=True),
                gr.update(visible=True)
            )

        good_button = gr.Button(visible=False, elem_id="good-button")
        bad_button = gr.Button(visible=False, elem_id="bad-button")

        good_button.click(fn=good_selected, outputs=[ai_message, tools_row])
        bad_button.click(fn=bad_selected, outputs=[ai_message, tools_row])

        voice_input.change(fn=handle_voice_input, inputs=voice_input, outputs=voice_output)

        # 이미지 업로드 시 자동 실행
        image_input.change(
            fn=classify_and_explain,
            inputs=image_input,
            outputs=[result, howto, tts_path_state]
        )

        # 음성 재생 버튼 클릭 시 실행
        play_button.click(
            fn=lambda path: path,
            inputs=tts_path_state,
            outputs=audio_output
        )

# Gradio 앱 실행
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7870)


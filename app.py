import streamlit as st
import os
import tempfile
import requests
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import time
import hashlib
from datetime import datetime
import re

# 필수 라이브러리들
try:
    import google.generativeai as genai
    from PyPDF2 import PdfReader
    from docx import Document
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    import numpy as np
    LIBS_AVAILABLE = True
except ImportError as e:
    st.error(f"라이브러리 설치 필요: {e}")
    LIBS_AVAILABLE = False

# 페이지 설정 (다크모드 강화)
st.set_page_config(
    page_title="AHN'S Advanced RAG Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 다크모드 CSS (Streamlit Cloud 호환)
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        color: #ffffff;
    }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .main-title {
        font-size: 2.5em;
        font-weight: bold;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin: 0;
    }
    
    .main-subtitle {
        font-size: 1.2em;
        color: #e0e0e0;
        margin-top: 0.5rem;
    }
    
    .chat-container {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .assistant-message {
        background: rgba(255,255,255,0.1);
        color: #ffffff;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .status-box {
        background: rgba(102, 126, 234, 0.2);
        border: 1px solid #667eea;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Streamlit 컴포넌트 스타일링 */
    .stTextInput > div > div > input {
        background-color: rgba(255,255,255,0.1);
        color: #ffffff;
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 10px;
    }
    
    .stSelectbox > div > div > select {
        background-color: rgba(255,255,255,0.1);
        color: #ffffff;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* 사이드바 스타일링 */
    .css-1d391kg, .css-1lcbmhc {
        background-color: rgba(30, 30, 46, 0.95);
    }
    
    .sidebar-content {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🛡️ AHN'S Advanced RAG Assistant</h1>
    <p class="main-subtitle">지능형 문서 분석 및 질의응답 시스템</p>
</div>
""", unsafe_allow_html=True)

class AdvancedRAGSystem:
    def __init__(self):
        """고급 RAG 시스템 (안전한 초기화)"""
        self.documents = []
        self.chunks = []
        self.embeddings = []
        self.is_fitted = False
        self.embedding_model = None
        
        # SentenceTransformer 모델 안전하게 로드
        try:
            with st.spinner('🤖 AI 모델 로딩 중...'):
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            st.success('✅ AI 모델 로드 완료!')
        except Exception as e:
            st.error(f"❌ SentenceTransformer 로드 실패: {e}")
            st.info("💡 TF-IDF 백업 모드로 전환합니다.")
            # TF-IDF 백업 모드
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
            self.use_tfidf_backup = True
        else:
            self.use_tfidf_backup = False
        
    def add_document(self, content: str, metadata: dict = None):
        """문서 추가"""
        doc_id = len(self.documents)
        self.documents.append({
            'id': doc_id,
            'content': content,
            'metadata': metadata or {}
        })
        
        # 청킹
        chunks = self.chunk_text(content, chunk_size=500, overlap=100)
        for i, chunk in enumerate(chunks):
            self.chunks.append({
                'doc_id': doc_id,
                'chunk_id': f"{doc_id}_{i}",
                'content': chunk,
                'metadata': metadata or {}
            })
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """텍스트를 청크로 분할"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 문장 경계에서 자르기
            if end < len(text):
                while end > start and text[end] not in '.!?\n':
                    end -= 1
                if end == start:
                    end = start + chunk_size
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
                
        return chunks
    
    def fit_vectors(self):
        """벡터화 수행 (안전한 처리)"""
        if not self.chunks:
            return False
            
        try:
            chunk_texts = [chunk['content'] for chunk in self.chunks]
            
            if not self.use_tfidf_backup and self.embedding_model is not None:
                # SentenceTransformer 사용
                self.embeddings = self.embedding_model.encode(chunk_texts)
            else:
                # TF-IDF 백업 모드
                self.embeddings = self.tfidf_vectorizer.fit_transform(chunk_texts)
            
            self.is_fitted = True
            return True
        except Exception as e:
            st.error(f"벡터화 오류: {e}")
            return False
    
    def hybrid_search(self, query: str, top_k: int = 3, alpha: float = 0.7) -> List[Dict]:
        """검색 수행 (안전한 처리)"""
        if not self.is_fitted or not self.chunks:
            return []
        
        try:
            if not self.use_tfidf_backup and self.embedding_model is not None:
                # SentenceTransformer 사용
                query_embedding = self.embedding_model.encode([query])
                similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
            else:
                # TF-IDF 백업 모드
                query_vector = self.tfidf_vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, self.embeddings).flatten()
            
            # 상위 결과 선택
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # 최소 유사도 임계값
                    results.append({
                        'chunk': self.chunks[idx],
                        'score': float(similarities[idx]),
                        'content': self.chunks[idx]['content']
                    })
            
            return results
            
        except Exception as e:
            st.error(f"검색 오류: {e}")
            return [])[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # 최소 유사도 임계값
                    results.append({
                        'chunk': self.chunks[idx],
                        'score': float(similarities[idx]),
                        'content': self.chunks[idx]['content']
                    })
            
            return results
            
        except Exception as e:
            st.error(f"검색 오류: {e}")
            return []

# 파일 처리 함수들
def extract_text_from_pdf(file) -> str:
    """PDF에서 텍스트 추출"""
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"PDF 처리 오류: {e}")
        return ""

def extract_text_from_docx(file) -> str:
    """DOCX에서 텍스트 추출"""
    try:
        doc = Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"DOCX 처리 오류: {e}")
        return ""

def extract_text_from_txt(file) -> str:
    """TXT 파일에서 텍스트 추출"""
    try:
        text = file.read().decode('utf-8')
        return text
    except Exception as e:
        st.error(f"TXT 처리 오류: {e}")
        return ""

# Luxia API 설정
LUXIA_API_KEY = "U2FsdGVkX19ZW0c+KOFb9zDy5eoyiz+I6icUKb2uOjuvUnzY1TaixWa5Ouy0s87vCdtqiQMmScIWcRbEJWcfXt/jS6RMWCW+38TU47bpj82JdafHt3ODi9VHfPmSrZJCMTwP4BJ471NZTqTLakFLpMQ/PTjafRebBJpfLSDeyBj4fX1VM+NnoH8u8aGG5AV4"

def get_luxia_response(prompt: str, context: str = "") -> str:
    """Luxia API 호출"""
    try:
        url = "https://api.luxia.one/api/luxia-chatbot-msg"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0"
        }
        
        full_prompt = f"{context}\n\n사용자 질문: {prompt}" if context else prompt
        
        payload = {
            "message": full_prompt,
            "key": LUXIA_API_KEY
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result.get('message', '응답을 받을 수 없습니다.')
        else:
            return f"API 오류 (상태 코드: {response.status_code})"
            
    except Exception as e:
        return f"API 호출 오류: {str(e)}"

def get_gemini_response(prompt: str, context: str = "") -> str:
    """Gemini API 호출 (백업)"""
    try:
        genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", ""))
        model = genai.GenerativeModel('gemini-pro')
        
        full_prompt = f"다음 문서를 참고하여 질문에 답해주세요:\n\n{context}\n\n질문: {prompt}"
        
        response = model.generate_content(full_prompt)
        return response.text
        
    except Exception as e:
        return f"Gemini API 오류: {str(e)}"

# 자동 문서 로드 함수
def load_default_document():
    """기본 문서 자동 로드"""
    # 실제 파일이 있는지 확인
    if os.path.exists("pstorm_pw.docx"):
        try:
            with open("pstorm_pw.docx", "rb") as f:
                content = extract_text_from_docx(f)
                if content.strip():
                    return content, "pstorm_pw.docx"
        except Exception as e:
            st.warning(f"기본 문서 로드 실패: {e}")
    
    # 하드코딩된 샘플 문서
    return """
# 회사 보안 정책 및 비밀번호 관리 가이드

## 1. 비밀번호 정책
- 최소 8자 이상, 영문 대소문자, 숫자, 특수문자 포함
- 90일마다 변경 필수
- 이전 5개 비밀번호 재사용 금지
- 개인정보 포함 금지 (생년월일, 이름 등)

## 2. 시스템 접근 보안
- 업무용 계정과 개인 계정 분리 사용
- 공용 컴퓨터에서 자동 로그인 설정 금지
- 업무 종료 시 반드시 화면 잠금
- USB 등 외부 저장매체 사용 시 보안 승인 필요

## 3. VPN 및 원격 접속
- 재택근무 시 회사 승인 VPN만 사용
- 공용 Wi-Fi에서 업무 시스템 접속 금지
- VPN 연결 시 개인용 프로그램 동시 사용 제한

## 4. 데이터 보호
- 회사 기밀 정보 개인 저장소 보관 금지
- 클라우드 서비스 이용 시 IT팀 승인 필요
- 정기 백업 수행 및 복구 테스트 실시

## 5. 보안 사고 대응
- 보안 사고 발견 시 즉시 IT보안팀 신고 (내선: 1588)
- 의심스러운 메일 수신 시 첨부파일 실행 금지
- 개인정보 유출 의심 시 개인정보보호팀 연락 (내선: 1577)

## 6. 교육 및 점검
- 분기별 보안 교육 이수 의무
- 월 1회 보안 점검 실시
- 보안 위반 시 경고 조치 및 재교육

## 연락처
- IT보안팀: 1588
- 개인정보보호팀: 1577
- 총무팀: 1500
""", "기본_보안정책.txt"

# 세션 상태 초기화
def initialize_session_state():
    """세션 상태 초기화"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = AdvancedRAGSystem()
        
        # 기본 문서 자동 로드
        default_content, filename = load_default_document()
        st.session_state.rag_system.add_document(
            default_content, 
            {'filename': filename, 'upload_time': datetime.now()}
        )
        st.session_state.rag_system.fit_vectors()
        
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []

# 메인 함수
def main():
    if not LIBS_AVAILABLE:
        st.error("필수 라이브러리가 설치되지 않았습니다. requirements.txt를 확인해주세요.")
        return
    
    initialize_session_state()
    
    # 사이드바 설정
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        st.markdown("### ⚙️ 설정")
        
        # AI 모델 선택
        ai_model = st.selectbox(
            "🧠 AI 모델 선택",
            ["Luxia", "Gemini"],
            index=0,
            help="응답 생성에 사용할 AI 모델을 선택하세요"
        )
        
        # API 키 설정
        st.markdown("### 🔑 Luxia API 키")
        luxia_key_display = "•" * 20 + LUXIA_API_KEY[-10:] if len(LUXIA_API_KEY) > 10 else "•" * 10
        st.text_input("API 키", value=luxia_key_display, disabled=True, type="password")
        
        st.markdown("### 🔍 검색 설정")
        
        # 검색 문서 수
        retrieval_count = st.slider(
            "검색할 문서 수",
            min_value=1,
            max_value=10,
            value=5,
            help="검색 시 참고할 문서 청크 개수"
        )
        
        # 하이브리드 검색 가중치
        alpha = st.slider(
            "하이브리드 검색 가중치",
            min_value=0.00,
            max_value=1.00,
            value=0.70,
            step=0.05,
            help="의미적 검색과 키워드 검색의 비율 조정"
        )
        
        st.markdown("### 📄 문서 업로드")
        
        # 파일 업로드
        uploaded_files = st.file_uploader(
            "문서를 업로드하세요",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="PDF, DOCX, TXT 파일을 업로드할 수 있습니다"
        )
        
        # 파일 처리
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in [f['name'] for f in st.session_state.processed_files]:
                    with st.spinner(f'{uploaded_file.name} 처리 중...'):
                        content = ""
                        
                        if uploaded_file.type == "application/pdf":
                            content = extract_text_from_pdf(uploaded_file)
                        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            content = extract_text_from_docx(uploaded_file)
                        elif uploaded_file.type == "text/plain":
                            content = extract_text_from_txt(uploaded_file)
                        
                        if content.strip():
                            st.session_state.rag_system.add_document(
                                content,
                                {
                                    'filename': uploaded_file.name,
                                    'upload_time': datetime.now(),
                                    'size': len(content)
                                }
                            )
                            st.session_state.rag_system.fit_vectors()
                            st.session_state.processed_files.append({
                                'name': uploaded_file.name,
                                'size': len(content),
                                'time': datetime.now()
                            })
                            st.success(f'{uploaded_file.name} 처리 완료!')
        
        # 시스템 상태
        st.markdown("### 📊 시스템 상태")
        
        # 상태 정보
        total_docs = len(st.session_state.rag_system.documents)
        total_chunks = len(st.session_state.rag_system.chunks)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1.5em;">📚</div>
                <div style="font-size: 1.2em; font-weight: bold;">{total_docs}</div>
                <div style="font-size: 0.9em;">로드된 문서</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1.5em;">🔍</div>
                <div style="font-size: 1.2em; font-weight: bold;">{total_chunks}</div>
                <div style="font-size: 0.9em;">청크</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 사용 중인 모델 정보
        st.markdown(f"""
        <div class="status-box">
            <strong>🧠 사용 중인 모델:</strong><br>
            {ai_model}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 메인 콘텐츠 영역
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # 질문 입력
    user_input = st.text_input(
        "질문을 입력하세요",
        placeholder="예: 비밀번호 정책이 무엇인가요?",
        key="user_question"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🔍 질문", type="primary", use_container_width=True)
    
    if ask_button and user_input:
        with st.spinner('답변 생성 중...'):
            # 검색 수행
            search_results = st.session_state.rag_system.hybrid_search(
                user_input,
                top_k=retrieval_count,
                alpha=alpha
            )
            
            # 컨텍스트 구성
            context = ""
            if search_results:
                context = "관련 문서 내용:\n\n"
                for i, result in enumerate(search_results, 1):
                    context += f"[문서 {i}]\n{result['content']}\n\n"
            
            # AI 응답 생성
            if ai_model == "Luxia":
                response = get_luxia_response(user_input, context)
            else:
                response = get_gemini_response(user_input, context)
            
            # 채팅 기록에 추가
            st.session_state.chat_history.append({
                'user': user_input,
                'assistant': response,
                'timestamp': datetime.now(),
                'search_results': search_results
            })
    
    # 채팅 기록 표시
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"""
        <div class="user-message">
            <strong>👤 질문:</strong> {chat['user']}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="assistant-message">
            <strong>🤖 답변:</strong><br>
            {chat['assistant']}
        </div>
        """, unsafe_allow_html=True)
        
        # 검색 결과 표시
        if chat.get('search_results'):
            with st.expander(f"📋 참고 문서 ({len(chat['search_results'])}개)"):
                for i, result in enumerate(chat['search_results'], 1):
                    st.markdown(f"""
                    **문서 {i}** (유사도: {result['score']:.3f})
                    
                    {result['content'][:300]}...
                    """)
        
        st.markdown("---")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 처리된 파일 목록
    if st.session_state.processed_files:
        st.markdown("### 📁 업로드된 파일")
        for file_info in st.session_state.processed_files:
            st.markdown(f"""
            <div class="status-box">
                📄 **{file_info['name']}**<br>
                크기: {file_info['size']:,} 글자 | 
                업로드: {file_info['time'].strftime('%Y-%m-%d %H:%M')}
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
import streamlit as st
import google.generativeai as genai
import PyPDF2
from docx import Document
import io
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import requests
import tempfile
import os
import re
from typing import List, Tuple, Dict
import json

# Streamlit 페이지 설정
st.set_page_config(
    page_title="Advanced RAG Assistant v2",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 다크모드 CSS 스타일링 (Luxia-ON 스타일)
def apply_dark_theme():
    st.markdown("""
    <style>
    /* 전체 배경 */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: #ffffff;
    }
    
    /* 메인 컨테이너 */
    .main-container {
        background: rgba(0, 0, 0, 0.7);
        border-radius: 20px;
        padding: 30px;
        margin: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* 헤더 스타일 */
    .header-container {
        text-align: center;
        margin-bottom: 30px;
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(45deg, #00d4ff, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .app-subtitle {
        color: #a0a0a0;
        font-size: 1.1rem;
        margin-bottom: 20px;
    }
    
    /* 채팅 컨테이너 */
    .chat-container {
        background: rgba(30, 30, 30, 0.9);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
    }
    
    /* 메시지 스타일 */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        margin-left: 20%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .bot-message {
        background: rgba(50, 50, 50, 0.9);
        color: #ffffff;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        margin-right: 20%;
        border-left: 4px solid #00d4ff;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    /* 입력 영역 */
    .input-container {
        background: rgba(30, 30, 30, 0.8);
        border-radius: 25px;
        padding: 10px;
        margin: 20px 0;
        border: 2px solid rgba(0, 212, 255, 0.3);
    }
    
    /* 사이드바 */
    .css-1d391kg {
        background: rgba(20, 20, 20, 0.9);
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* 선택박스 */
    .stSelectbox > div > div {
        background: rgba(50, 50, 50, 0.8);
        color: white;
        border-radius: 10px;
    }
    
    /* 파일 업로더 */
    .stFileUploader > div {
        background: rgba(50, 50, 50, 0.8);
        border-radius: 15px;
        border: 2px dashed rgba(0, 212, 255, 0.5);
    }
    
    /* 메트릭 카드 */
    .metric-card {
        background: rgba(50, 50, 50, 0.8);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #00d4ff;
    }
    
    /* 스크롤바 */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* 숨겨야 할 요소들 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 전역 변수
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'tfidf_matrix' not in st.session_state:
    st.session_state.tfidf_matrix = None

class AdvancedRAGSystem:
    def __init__(self):
        self.embedding_model = None
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words=None
        )
        
    def load_embedding_model(self):
        """임베딩 모델 로드"""
        if self.embedding_model is None:
            with st.spinner("🔄 임베딩 모델 로딩 중..."):
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.embedding_model
    
    def chunk_text_with_overlap(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """오버랩이 있는 텍스트 청킹"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 문장 경계에서 자르기
            if end < len(text):
                # 마지막 문장 끝을 찾기
                last_period = text.rfind('.', start, end)
                last_exclamation = text.rfind('!', start, end)
                last_question = text.rfind('?', start, end)
                
                sentence_end = max(last_period, last_exclamation, last_question)
                
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 오버랩 고려한 다음 시작점
            start = end - overlap if end < len(text) else len(text)
            
            if start >= len(text):
                break
                
        return chunks
    
    def hybrid_search(self, query: str, documents: List[str], embeddings: np.ndarray, 
                     top_k: int = 5, alpha: float = 0.7) -> List[Tuple[str, float]]:
        """하이브리드 검색 (벡터 + 키워드)"""
        if not documents:
            return []
        
        # 벡터 검색
        query_embedding = self.embedding_model.encode([query])
        vector_similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # 키워드 검색 (TF-IDF)
        try:
            query_tfidf = self.vectorizer.transform([query])
            keyword_similarities = cosine_similarity(query_tfidf, st.session_state.tfidf_matrix)[0]
        except:
            keyword_similarities = np.zeros(len(documents))
        
        # 하이브리드 점수 계산
        hybrid_scores = alpha * vector_similarities + (1 - alpha) * keyword_similarities
        
        # 상위 문서 선택
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if hybrid_scores[idx] > 0:
                results.append((documents[idx], hybrid_scores[idx]))
        
        return results
    
    def rerank_documents(self, query: str, documents: List[Tuple[str, float]], 
                        max_docs: int = 3) -> List[str]:
        """Re-ranking with query relevance"""
        if not documents:
            return []
        
        # 간단한 키워드 기반 re-ranking
        query_terms = set(query.lower().split())
        
        reranked = []
        for doc, score in documents:
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms.intersection(doc_terms))
            
            # 키워드 오버랩과 기존 점수 결합
            final_score = score + (overlap * 0.1)
            reranked.append((doc, final_score))
        
        # 최종 점수로 정렬
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in reranked[:max_docs]]

# RAG 시스템 초기화
rag_system = AdvancedRAGSystem()

def setup_luxia_api(api_key: str):
    """Luxia API 설정"""
    return api_key

def setup_gemini_api(api_key: str):
    """Gemini API 설정"""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-pro')

def generate_luxia_response(prompt: str, context: str, api_key: str) -> str:
    """Luxia API 호출"""
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        if context:
            full_prompt = f"""문서 내용을 참고하여 질문에 답변해주세요.

문서 내용:
{context}

질문: {prompt}

답변:"""
        else:
            full_prompt = prompt
        
        data = {
            'model': 'luxia-2.5',
            'messages': [
                {'role': 'user', 'content': full_prompt}
            ],
            'max_tokens': 1000,
            'temperature': 0.7
        }
        
        # Luxia API 엔드포인트 (실제 API 명세에 따라 수정 필요)
        response = requests.post(
            'https://api.saltlux.com/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"❌ Luxia API 오류: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"❌ Luxia API 연결 오류: {str(e)}"

def generate_gemini_response(prompt: str, context: str, model) -> str:
    """Gemini API 호출"""
    try:
        if context:
            full_prompt = f"""문서 내용을 참고하여 질문에 답변해주세요.

문서 내용:
{context}

질문: {prompt}

답변:"""
        else:
            full_prompt = prompt
        
        response = model.generate_content(full_prompt)
        return response.text
        
    except Exception as e:
        return f"❌ Gemini API 오류: {str(e)}"

def extract_text_from_pdf(pdf_file):
    """PDF에서 텍스트 추출"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"PDF 처리 오류: {e}")
        return ""

def extract_text_from_docx(docx_file):
    """DOCX에서 텍스트 추출"""
    try:
        doc = Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"DOCX 처리 오류: {e}")
        return ""

def process_documents(uploaded_files):
    """업로드된 문서들 처리"""
    all_chunks = []
    
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
        else:
            text = str(uploaded_file.read(), "utf-8")
        
        if text.strip():
            # 오버랩이 있는 청킹
            chunks = rag_system.chunk_text_with_overlap(
                text, 
                chunk_size=500, 
                overlap=50
            )
            all_chunks.extend(chunks)
    
    return all_chunks

def main():
    # 다크 테마 적용
    apply_dark_theme()
    
    # 헤더
    st.markdown("""
    <div class="header-container">
        <div class="app-title">🤖 Advanced RAG Assistant v2</div>
        <div class="app-subtitle">Luxia & Gemini 기반 지능형 문서 분석 시스템</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바 설정
    with st.sidebar:
        st.markdown("### ⚙️ 설정")
        
        # AI 모델 선택
        ai_model = st.selectbox(
            "🤖 AI 모델 선택",
            ["Luxia", "Gemini"],
            index=0,  # 기본값: Luxia
            help="사용할 AI 모델을 선택하세요"
        )
        
        # API 키 입력
        if ai_model == "Luxia":
            api_key = st.text_input(
                "🔑 Luxia API 키",
                value="U2FsdGVkX19ZW0c+KOFb9zDy5eoyiz+I6icUKb2uOjuvUnzY1TaixWa5Ouy0s87vCdtqiQMmScIWcRbEJWcfXt/jS6RMWCW+38TU47bpj82JdafHt3ODi9VHfPmSrZJCMTwP4BJ471NZTqTLakFLpMQ/PTjafRebBJpfLSDeyBj4fX1VM+NnoH8u8aGG5AV4",
                type="password"
            )
        else:
            api_key = st.text_input(
                "🔑 Gemini API 키",
                type="password"
            )
        
        st.markdown("---")
        
        # 검색 설정
        st.markdown("### 🔍 검색 설정")
        
        search_k = st.slider(
            "검색할 문서 수",
            min_value=1,
            max_value=10,
            value=5,
            help="유사한 문서를 몇 개까지 검색할지 설정"
        )
        
        alpha = st.slider(
            "하이브리드 검색 가중치",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="벡터 검색(1.0) vs 키워드 검색(0.0)"
        )
        
        st.markdown("---")
        
        # 문서 업로드
        st.markdown("### 📄 문서 업로드")
        uploaded_files = st.file_uploader(
            "문서를 업로드하세요",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="PDF, DOCX, TXT 파일을 지원합니다"
        )
        
        if uploaded_files:
            if st.button("📊 문서 처리", type="primary"):
                with st.spinner("🔄 문서 처리 중..."):
                    # 문서 처리
                    chunks = process_documents(uploaded_files)
                    st.session_state.documents = chunks
                    
                    if chunks:
                        # 임베딩 모델 로드
                        embedding_model = rag_system.load_embedding_model()
                        
                        # 벡터 임베딩 생성
                        embeddings = embedding_model.encode(chunks)
                        st.session_state.embeddings = embeddings
                        
                        # TF-IDF 행렬 생성
                        st.session_state.tfidf_matrix = rag_system.vectorizer.fit_transform(chunks)
                        
                        st.success(f"✅ {len(chunks)}개 청크로 문서 처리 완료!")
                        
                        # 문서 통계
                        st.markdown("### 📊 문서 통계")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("총 청크 수", len(chunks))
                        with col2:
                            avg_length = sum(len(chunk) for chunk in chunks) // len(chunks)
                            st.metric("평균 청크 길이", f"{avg_length}자")
    
    # 메인 컨텐츠
    if not api_key:
        st.warning("⚠️ API 키를 입력해주세요.")
        return
    
    # API 설정
    if ai_model == "Luxia":
        model = setup_luxia_api(api_key)
    else:
        model = setup_gemini_api(api_key)
        st.session_state.model = model
    
    # 채팅 인터페이스
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # 대화 기록 표시
    for i, (role, message) in enumerate(st.session_state.conversation):
        if role == "user":
            st.markdown(f'<div class="user-message">👤 {message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">🤖 {message}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 입력 영역
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "",
            placeholder="질문을 입력하세요...",
            key="user_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("전송", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 메시지 처리
    if send_button and user_input:
        # 사용자 메시지 추가
        st.session_state.conversation.append(("user", user_input))
        
        # 문서 검색
        context = ""
        if st.session_state.documents and st.session_state.embeddings is not None:
            with st.spinner("🔍 관련 문서 검색 중..."):
                # 하이브리드 검색
                search_results = rag_system.hybrid_search(
                    user_input,
                    st.session_state.documents,
                    st.session_state.embeddings,
                    top_k=search_k,
                    alpha=alpha
                )
                
                # Re-ranking
                relevant_docs = rag_system.rerank_documents(user_input, search_results, max_docs=3)
                context = "\n\n".join(relevant_docs)
        
        # AI 응답 생성
        with st.spinner(f"🤖 {ai_model} 응답 생성 중..."):
            if ai_model == "Luxia":
                response = generate_luxia_response(user_input, context, api_key)
            else:
                response = generate_gemini_response(user_input, context, st.session_state.model)
        
        # 봇 응답 추가
        st.session_state.conversation.append(("bot", response))
        
        # 페이지 새로고침
        st.rerun()
    
    # 하단 정보
    if st.session_state.documents:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📚 로드된 문서", f"{len(st.session_state.documents)}개 청크")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🤖 사용 중인 모델", ai_model)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("💬 대화 수", len(st.session_state.conversation) // 2)
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
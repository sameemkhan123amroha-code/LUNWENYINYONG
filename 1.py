import streamlit as st
import pandas as pd
from pypdf import PdfReader
from io import BytesIO
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

# --- 1. 页面配置 ---
st.set_page_config(page_title="全能论文助手 (智谱修复版)", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 20px; }
    .stButton>button { width: 100%; border-radius: 6px; height: 3em; font-weight: 600; }
    .interactive-sent { cursor: pointer; border-bottom: 2px solid #e0e0e0; padding: 0 2px; transition: all 0.2s; line-height: 1.8; }
    .interactive-sent:hover { background-color: #fff8e1; border-bottom-color: #ffc107; }
    .info-tooltip { display: none; position: fixed; background: #ffffff; border: 1px solid #d1d5da; box-shadow: 0 10px 30px rgba(0,0,0,0.15); padding: 16px; z-index: 999999; width: 400px; border-radius: 8px; font-family: sans-serif; font-size: 14px; line-height: 1.5; color: #24292e; }
    .tooltip-header { display: flex; justify-content: space-between; margin-bottom: 8px; border-bottom: 1px solid #eaecef; padding-bottom: 8px;}
    .tooltip-source { font-weight: 700; color: #0366d6; font-size: 13px; }
    .tooltip-score { font-weight: 700; font-size: 13px; }
    .tooltip-content { background: #f6f8fa; padding: 12px; border-radius: 6px; font-size: 13px; max-height: 200px; overflow-y: auto; color: #444; border: 1px solid #eaecef;}
    .spacer { height: 250px; }
</style>
""", unsafe_allow_html=True)

# --- 2. 核心逻辑函数 ---

# 尝试导入本地模型库（仅作为备用，不强制）
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = None

def get_pdf_text(pdf_docs):
    text_data = []
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                t = page.extract_text()
                if t: text += t
            text_data.append({"filename": pdf.name, "text": text})
        except Exception as e:
            st.error(f"⚠️ 文件 {pdf.name} 读取失败: {e}")
    return text_data

def get_vectorstore(text_data, use_online_embed, api_key, api_base, provider):
    """
    构建向量库
    """
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    for item in text_data:
        chunks = text_splitter.split_text(item["text"])
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={"source": item["filename"]}))
    
    embeddings = None
    
    # --- 核心修改：在线 Embedding 逻辑 ---
    if use_online_embed:
        if not api_key:
            st.error("❌ 使用在线 Embedding 需要提供 API Key！")
            st.stop()
        
        # 智能判断 Embedding 模型名称
        embed_model_name = "text-embedding-3-small" # OpenAI 默认
        if "Zhipu" in provider:
            embed_model_name = "embedding-2" # 智谱专用 Embedding 模型
        
        try:
            # 使用 LangChain 的 OpenAI 兼容接口调用智谱 Embedding
            # 💡【关键修复】：增加 chunk_size=16 参数
            # 智谱 API 限制单次请求最大 64 条，LangChain 默认是 1000，必须改成小于 64
            embeddings = OpenAIEmbeddings(
                openai_api_key=api_key, 
                openai_api_base=api_base,
                model=embed_model_name,
                chunk_size=16  # <--- 这里是修复 1214 错误的关键
            )
        except Exception as e:
            st.error(f"❌ 在线 Embedding 初始化失败: {e}")
            st.stop()
            
    else:
        # 本地模型逻辑 (备用)
        if HuggingFaceEmbeddings is None:
            st.error("❌ 缺少 sentence-transformers 库，无法使用本地模型。建议勾选上方 '使用在线 Embedding'。")
            st.stop()
        try:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # 国内镜像
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            st.error(f"❌ 本地模型加载失败: {e}。建议勾选上方 '使用在线 Embedding' 改用智谱接口。")
            st.stop()

    # 这里会触发批量的 Embedding 请求，chunk_size=16 会确保不超限
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore, embeddings

# --- 3. 界面布局 ---

st.title("🤖 智能论文助手 (智谱修复版)")

# --- 配置区 ---
with st.container():
    col_config, col_upload = st.columns([1, 1.2])
    
    with col_config:
        with st.expander("🛠️ 模型参数配置", expanded=True):
            provider = st.selectbox(
                "1. 选择大模型厂商",
                [
                    "Zhipu AI (智谱GLM)",   # 推荐
                    "DeepSeek (深度求索)", 
                    "OpenAI (GPT-4o)", 
                    "Kimi (月之暗面)", 
                    "Custom (自定义)"
                ]
            )
            
            # 预设参数
            p_url = "https://open.bigmodel.cn/api/paas/v4/"
            p_model = "glm-4"
            
            # 默认勾选在线 Embedding
            default_use_online = True 
            
            if "Zhipu" in provider:
                p_url = "https://open.bigmodel.cn/api/paas/v4/"
                p_model = "glm-4"
                default_use_online = True # 智谱默认使用在线，省去本地麻烦
            elif "DeepSeek" in provider:
                p_url = "https://api.deepseek.com/v1" # 自动修正 /v1
                p_model = "deepseek-chat"
                default_use_online = False # DeepSeek 没有 embedding，默认走本地
            elif "OpenAI" in provider:
                p_url = "https://api.openai.com/v1"
                p_model = "gpt-4o"
                default_use_online = True
            elif "Kimi" in provider:
                p_url = "https://api.moonshot.cn/v1"
                p_model = "moonshot-v1-8k"
                default_use_online = False
            
            api_base = st.text_input("Base URL", value=p_url)
            model_name = st.text_input("Model Name", value=p_model)
            api_key = st.text_input("API Key", type="password", placeholder="输入 Key...")

            st.markdown("---")
            
            # 关键复选框
            use_online_embed = st.checkbox(
                "使用在线 Embedding (智谱/OpenAI 用户强烈推荐勾选)", 
                value=default_use_online,
                help="勾选后将使用厂商的 API 进行向量化，无需下载本地模型。智谱用户请务必勾选！"
            )
            
            if use_online_embed and "Zhipu" in provider:
                st.caption("✅ 已启用智谱 `embedding-2`，已修复 64 条限制问题。")

    with col_upload:
        uploaded_files = st.file_uploader("📂 导入 PDF 论文", accept_multiple_files=True, type=['pdf'])
        if uploaded_files:
            if 'processed_data' not in st.session_state or st.session_state.get('file_count') != len(uploaded_files):
                with st.spinner("📄 解析 PDF 中..."):
                    st.session_state.processed_data = get_pdf_text(uploaded_files)
                    st.session_state.file_count = len(uploaded_files)
                st.success(f"✅ 已加载 {len(uploaded_files)} 篇论文")

st.divider()

# --- Excel 整理 ---
if st.session_state.get('processed_data'):
    col_btn, col_dl = st.columns([1, 4])
    with col_btn:
        do_excel = st.button("📊 一键整理成 EXCEL")
    
    if do_excel:
        if not api_key:
            st.error("❌ 请先输入 API Key")
        else:
            with st.spinner(f"正在使用 {model_name} 阅读并总结..."):
                try:
                    llm = ChatOpenAI(base_url=api_base, api_key=api_key, model=model_name, temperature=0.1)
                    summary_list = []
                    prog = st.progress(0)
                    total = len(st.session_state.processed_data)
                    for i, item in enumerate(st.session_state.processed_data):
                        prompt = f"任务：用中文概括这篇论文的核心内容。\n【限制】：严格控制在 20 个汉字以内！直接写结论。\n论文片段：{item['text'][:2000]}"
                        res = llm.invoke(prompt)
                        summary_list.append({"论文名称": item["filename"], "论文大致意思": res.content.strip()})
                        prog.progress((i+1)/total)
                    
                    df = pd.DataFrame(summary_list)
                    out = BytesIO()
                    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False)
                        writer.sheets['Sheet1'].set_column('A:B', 40)
                    
                    with col_dl:
                        st.download_button("⬇️ 下载 Excel", out.getvalue(), "论文整理.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    st.dataframe(df, height=200)
                except Exception as e:
                    st.error(f"API 错误: {e}")

st.divider()

# --- 综述生成 ---
st.subheader("📝 智能综述生成 (RAG + 语义核查)")

c1, c2 = st.columns([4, 1])
query = c1.text_input("输入提示词", placeholder="例如：地聚物的抗压强度影响因素")
start_rag = c2.button("🚀 生成综述")

if start_rag:
    if not api_key:
        st.error("❌ 请先输入 API Key")
    elif not st.session_state.get('processed_data'):
        st.warning("⚠️ 请先上传 PDF")
    elif not query:
        st.warning("⚠️ 请输入提示词")
    else:
        with st.spinner("🔍 正在检索资料..."):
            try:
                # 1. 检索 (传入 use_online_embed 参数)
                vectorstore, embed_model = get_vectorstore(
                    st.session_state.processed_data, 
                    use_online_embed, 
                    api_key, 
                    api_base, 
                    provider
                )
                
                docs = vectorstore.similarity_search(query, k=8)
                
                if not docs:
                    st.error("未找到相关内容。")
                    st.stop()
                
                context = "\n".join([f"【来源:{d.metadata['source']}】{d.page_content}" for d in docs])
                
                # 2. 生成
                with st.spinner(f"✍️ 正在使用 {model_name} 撰写综述..."):
                    llm_chat = ChatOpenAI(base_url=api_base, api_key=api_key, model=model_name, temperature=0.3)
                    sys_prompt = f"你是一个学术助手。基于以下资料撰写关于“{query}”的综述。\n要求：\n1. 忠实于原文。\n2. 句尾标注来源 (文件名)。\n3. 输出纯 HTML (不含 <html>)，分段落 <p>。\n资料：\n{context}"
                    resp = llm_chat.invoke(sys_prompt)
                    raw_html = resp.content.replace("```html", "").replace("```", "")
                
                # 3. 语义核查
                sentences = re.split(r'(?<=[。！？])', raw_html)
                html_parts = []
                stat = st.empty()
                
                for idx, sent in enumerate(sentences):
                    clean = re.sub(r'<[^>]+>', '', sent).strip()
                    if len(clean) < 5:
                        html_parts.append(sent)
                        continue
                    
                    evidence_docs = vectorstore.similarity_search(clean, k=1)
                    if evidence_docs:
                        doc = evidence_docs[0]
                        v1 = embed_model.embed_query(clean)
                        v2 = embed_model.embed_query(doc.page_content)
                        score = cosine_similarity([v1], [v2])[0][0] * 100
                        
                        safe_txt = doc.page_content[:300].replace('"', '&quot;').replace('\n', ' ')
                        span = f"""<span id="s_{idx}" class="interactive-sent" onclick="showTip('s_{idx}', '{doc.metadata['source']}', {round(score,1)}, '{safe_txt}')">{sent}</span>"""
                        html_parts.append(span)
                    else:
                        html_parts.append(sent)
                
                stat.empty()
                full_html = "".join(html_parts) + "<div class='spacer'></div>"
                
                js = """
                <div id="tip" class="info-tooltip">
                    <div class="tooltip-header"><span id="t-src" class="tooltip-source"></span><span id="t-score" class="tooltip-score"></span></div>
                    <div style="font-weight:bold;margin-bottom:5px">语义证据:</div>
                    <div id="t-txt" class="tooltip-content"></div>
                </div>
                <script>
                function showTip(id, src, sc, txt) {
                    var t = document.getElementById('tip');
                    var el = document.getElementById(id);
                    var r = el.getBoundingClientRect();
                    var scrollTop = window.pageYOffset || document.documentElement.scrollTop;
                    var scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;
                    document.getElementById('t-src').innerText = '📄 ' + src;
                    document.getElementById('t-score').innerHTML = '匹配度: <span style="color:' + (sc>75?'#2da44e':sc>60?'#d29922':'#cf222e') + '">' + sc + '%</span>';
                    document.getElementById('t-txt').innerHTML = txt;
                    t.style.display = 'block';
                    t.style.top = (scrollTop + r.bottom + 5) + 'px';
                    t.style.left = (scrollLeft + r.left) + 'px';
                    setTimeout(() => { document.addEventListener('click', function c(e) {
                        if(e.target.id !== id && !t.contains(e.target)) { t.style.display = 'none'; document.removeEventListener('click', c); }
                    })}, 100);
                }
                </script>
                """
                
                st.success("✅ 生成完毕！")
                st.components.v1.html(f"<div style='font-family:sans-serif;padding:10px'>{full_html}</div>{js}", height=600, scrolling=True)
                
            except Exception as e:
                st.error(f"❌ 运行错误: {e}")
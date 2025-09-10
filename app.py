
# -*- coding: utf-8 -*-
import os, pickle, time
import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Try import mediapipe
try:
    import mediapipe as mp
    MP_OK = True
except Exception:
    MP_OK = False

st.set_page_config(page_title="Quem é seu gêmeo histórico", layout="wide")

# CSS styles including CSS grid background
st.markdown("""
<style>
:root{--bg:#0d0d0d; --gold:#FFD700; --muted:#2E2E2E; --light:#F5F5F5;}

body, html {background: var(--bg);}
.header{text-align:center; padding:20px 0;}
h1.title {color:var(--gold); font-family: 'Georgia', serif; text-shadow:0 0 12px rgba(255,215,0,0.8);}

/* Grid background */
.bg-grid {
  position: fixed;
  inset: 0;
  z-index: 0;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 8px;
  padding: 30px;
  opacity: 0.18;
  filter: grayscale(60%) blur(1px);
}
.bg-grid img{width:100%; height:140px; object-fit:cover; border:4px solid rgba(255,215,0,0.06); box-shadow:0 6px 12px rgba(0,0,0,0.6);}

/* Content container */
.content{position:relative; z-index:2;}

/* Buttons */
.btn-gold button{background:#000; color:var(--gold); border:2px solid var(--gold); padding:10px 18px; border-radius:10px;}
.btn-gold button:hover{background:var(--gold); color:#000; transform:scale(1.03); box-shadow:0 0 18px var(--gold);}

/* Cards */
.card{background:#111; border:2px solid var(--gold); border-radius:12px; padding:12px; text-align:center;}
.card img{width:100%; height:180px; object-fit:cover; border-radius:8px;}
.card h3{color:var(--gold);}

/* Museum grid */
.museum-grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:16px;}
.museum-item{background:#111; padding:8px; border-radius:8px; border:1px solid rgba(255,215,0,0.06); text-align:center;}
.museum-item img{width:100%; height:140px; object-fit:cover; border-radius:6px;}
</style>
""", unsafe_allow_html=True)

# Background grid by loading images from imagens folder
def build_bg_grid_html():
    imgs = []
    base = os.path.join('imagens')
    for gender in ['masculino','feminino']:
        p = os.path.join(base, gender)
        if not os.path.isdir(p): continue
        for fn in os.listdir(p):
            if fn.lower().endswith(('.jpg','.jpeg','.png')):
                imgs.append(os.path.join(p, fn).replace('\\','/'))
    imgs = imgs[:60]
    html = ['<div class="bg-grid">']
    for img in imgs:
        html.append(f'<img src="{img}" />')
    html.append('</div>')
    return '\n'.join(html)

st.markdown(build_bg_grid_html(), unsafe_allow_html=True)

st.markdown('<div class="content">', unsafe_allow_html=True)
st.markdown('<div class="header"><h1 class="title">Quem é seu gêmeo histórico</h1></div>', unsafe_allow_html=True)

# Session state
if 'page' not in st.session_state:
    st.session_state.page = 'intro'
if 'genero' not in st.session_state:
    st.session_state.genero = None
if 'last_winner' not in st.session_state:
    st.session_state.last_winner = None
if 'last_top3' not in st.session_state:
    st.session_state.last_top3 = None

# Embedding extraction using MediaPipe FaceMesh
def extract_embedding_from_pil(pil):
    if not MP_OK:
        return None
    img = cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2BGR)
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        coords = np.array([[p.x, p.y, p.z] for p in res.multi_face_landmarks[0].landmark], dtype=np.float32)
        center = coords.mean(axis=0, keepdims=True)
        coords -= center
        scale = np.linalg.norm(coords, axis=1).max()
        if scale == 0: return None
        coords /= scale
        return coords.flatten()

# Load or build embeddings DB (cached)
@st.cache_resource
def load_embeddings():
    emb_file = 'embeddings.pkl'
    if os.path.exists(emb_file):
        try:
            with open(emb_file,'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    data = {'masculino':{}, 'feminino':{}}
    base = 'imagens'
    for gender in ['masculino','feminino']:
        p = os.path.join(base, gender)
        if not os.path.isdir(p): continue
        for fn in sorted(os.listdir(p)):
            if not fn.lower().endswith(('.jpg','.jpeg','.png')): continue
            path = os.path.join(p, fn)
            try:
                pil = Image.open(path).convert('RGB')
            except Exception:
                continue
            emb = extract_embedding_from_pil(pil) if MP_OK else None
            if emb is not None:
                name = os.path.splitext(fn)[0]
                data[gender][name] = emb
    try:
        with open(emb_file,'wb') as f:
            pickle.dump(data,f)
    except Exception:
        pass
    return data

def top_k_matches(emb, db_dict, k=3):
    if emb is None or not db_dict:
        return []
    scores = []
    for name, vec in db_dict.items():
        a = emb / (np.linalg.norm(emb)+1e-9)
        b = vec / (np.linalg.norm(vec)+1e-9)
        cos = float(np.dot(a,b))
        sim = (cos+1.0)/2.0
        scores.append((name, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]

# Pages
# Intro page
if st.session_state.page == 'intro':
    st.write('')
    if os.path.exists(os.path.join('videos','intro.mp4')) and os.path.getsize(os.path.join('videos','intro.mp4'))>0:
        st.video(os.path.join('videos','intro.mp4'))
    else:
        st.info('Vídeo de introdução não encontrado. Clique em Continuar para prosseguir.')
    st.markdown('<div class="btn-gold">', unsafe_allow_html=True)
    if st.button('👉 Continuar'):
        st.session_state.page = 'genero'
        st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Gender selection
elif st.session_state.page == 'genero':
    st.markdown('<h2 style="text-align:center;">Escolha o gênero para comparar</h2>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="btn-gold">', unsafe_allow_html=True)
        if st.button('👨 Masculino', key='btn_m'):
            st.session_state.genero = 'masculino'; st.session_state.page='compare'; st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="btn-gold">', unsafe_allow_html=True)
        if st.button('👩 Feminino', key='btn_f'):
            st.session_state.genero = 'feminino'; st.session_state.page='compare'; st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Comparison page
elif st.session_state.page == 'compare':
    genero = st.session_state.genero or 'masculino'
    st.markdown(f'<h2>📸 Envie sua foto — grupo: <strong>{genero.title()}</strong></h2>', unsafe_allow_html=True)
    db = load_embeddings()
    uploaded = st.file_uploader('Selecione uma foto', type=['jpg','jpeg','png'])
    if uploaded:
        pil = Image.open(uploaded).convert('RGB')
        st.image(pil, caption='Sua foto', use_column_width=True)
        with st.spinner('Processando...'):
            emb = extract_embedding_from_pil(pil) if MP_OK else None
            if emb is None:
                st.error('Rosto não detectado ou MediaPipe não instalado.')
            else:
                top3 = top_k_matches(emb, db.get(genero, {}), k=3)
                if not top3:
                    st.info('Sem correspondências na base.')
                else:
                    winner = top3[0][0]
                    st.session_state.last_winner = winner
                    st.session_state.last_top3 = top3
                    # Play winner video first (suspense)
                    video_path = os.path.join('videos','resultados', f'{winner}.mp4')
                    if os.path.exists(video_path) and os.path.getsize(video_path)>0:
                        st.video(video_path)
                    else:
                        st.info('Vídeo do personagem não disponível. (placeholder)')
                    # Reveal button
                    if st.button('🎭 Revelar Resultado'):
                        # Show winner and top3
                        st.success(f'Seu gêmeo histórico é: {{winner.replace("_"," ").title()}}')
                        st.write('---')
                        st.subheader('🔎 Top 3 mais parecidos')
                        cols = st.columns(3)
                        for (name, score), col in zip(top3, cols):
                            with col:
                                st.markdown('<div class="card">', unsafe_allow_html=True)
                                img_path = None
                                for ext in ('.jpg','.jpeg','.png'):
                                    p = os.path.join('imagens', genero, f"{name}{ext}")
                                    if os.path.exists(p): img_path = p; break
                                if img_path and os.path.exists(img_path):
                                    st.image(img_path, use_column_width=True)
                                st.markdown(f"<h3>{name.replace('_',' ').title()}</h3>", unsafe_allow_html=True)
                                st.write(f"Similaridade: **{int(score*100)}%**")
                                st.markdown('</div>', unsafe_allow_html=True)
                        # Show museum button
                        if st.button('🏛️ Museu dos Personagens'):
                            st.session_state.page = 'museu'
                            st.experimental_rerun()

# Museum page
elif st.session_state.page == 'museu':
    st.markdown('<h1 style="text-align:center;">Museu dos Personagens</h1>', unsafe_allow_html=True)
    tab = st.radio('Ver:', ['Todos','Feminino','Masculino'], horizontal=True)
    base = os.path.join('imagens')
    items = []
    for gender in ['masculino','feminino']:
        p = os.path.join(base, gender)
        if not os.path.isdir(p): continue
        for fn in sorted(os.listdir(p)):
            if not fn.lower().endswith(('.jpg','.jpeg','.png')): continue
            sname = os.path.splitext(fn)[0]
            items.append((sname, os.path.join(p, fn).replace('\\','/'), gender))
    if tab == 'Feminino':
        items = [it for it in items if it[2]=='feminino']
    elif tab == 'Masculino':
        items = [it for it in items if it[2]=='masculino']
    st.markdown('<div class="museum-grid">', unsafe_allow_html=True)
    for name, path, gender in items:
        st.markdown(f'<div class="museum-item"><img src="{path}"/><div style="padding-top:6px;color:#f5f5f5;">{name.replace("_"," ").title()}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    if st.button('⬅️ Voltar ao início'):
        st.session_state.page = 'intro'
        st.experimental_rerun()

st.markdown('</div>', unsafe_allow_html=True)

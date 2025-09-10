[README.md](https://github.com/user-attachments/files/22259214/README.md)
# Quem é seu gêmeo histórico

## Visão geral
Site em Streamlit que compara seu rosto com 50 figuras históricas usando MediaPipe FaceMesh e mostra os 3 mais parecidos.
Fluxo: vídeo intro → escolha gênero → upload foto → vídeo do vencedor (suspense) → revelação e top 3 → museu.

## Como usar localmente
1. Instale dependências:
   pip install -r requirements.txt
2. (Opcional) Gere embeddings para acelerar (requer mediapipe):
   python process_faces.py
3. Rode o app:
   streamlit run app.py

## Deploy no Render
- Build command: pip install -r requirements.txt
- Start command: streamlit run app.py --server.port 10000 --server.address 0.0.0.0

## Estrutura do projeto
- app.py - aplicação principal
- process_faces.py - script opcional para pré-gerar embeddings
- imagens/masculino/ e imagens/feminino/ - fotos dos personagens
- videos/intro.mp4 - vídeo de introdução (placeholder)
- videos/resultados/ - 50 placeholders de vídeo (substitua pelos reais quando tiver)
- embeddings.pkl - (opcional) gerado automaticamente

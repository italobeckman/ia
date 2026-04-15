# Product Requirements Document (PRD) Minimalista
## 🧪 Sandbox Experimental: Object Detection com Hugging Face 

### 1. Visão Geral
O propósito deste documento é definir os requisitos mínimos para montar um **ambiente simples de testes local (sandbox)** voltado puramente para aprendizado e exploração. O sistema não é uma solução de produção, mas atuará como laboratório, possibilitando carregar imagens e visualizá-las rapidamente com as respectivas bounding boxes desenhadas por modelos da Hugging Face (como DETR, YOLO, entre outros).

### 2. Objetivos
- **Objetivo Central:** Entender o fluxo de uso da biblioteca `transformers` aplicado a tarefas de Visão Computacional de "Object Detection" (Detecção de Objetos).
- **Metas do Laboratório:**
  - Garantir o download transparente dos pesos dos modelos e o aquecimento (cache local).
  - Executar inferências locais em imagens de exemplo do usuário de forma ágil e iterativa.
  - Testar comparativamente as saídas de alguns modelos populares abertos na plataforma, verificando graus de acerto para propósitos de estudo.

### 3. Requisitos Funcionais (RF)
- **RF01 - Input de Imagem Estruturado:** O usuário poderá usar uma imagem estática local (passando apenas o caminho do arquivo `.jpg`/`.png`) ou fazer upload manual pela interface do ambiente.
- **RF02 - Execução Simplificada (Pipeline):** Utilização direta do atalho `pipeline("object-detection", model="...")` fornecido pela biblioteca nativa, enviando as imagens para inferência pelo modelo definido.
- **RF03 - Visualização Transparente (Output):** O ambiente deverá expor ao final do código três elementos:
  - A impressão dos dados estruturados brutos recebidos (lista de objetos e score de certeza).
  - O desenho impresso na tela da imagem junto com as caixas vermelhas nos devidos eixos X e Y.
  - Um relatório técnico textual gerado automaticamente após cada inferência (ver RF04).
- **RF04 - Relatório Técnico de Inferência:** Após cada detecção, o sistema deverá exibir um relatório estruturado contendo:
  - **Hardware utilizado** na inferência (CUDA GPU ou CPU).
  - **Total de objetos detectados** pelo modelo (bruto, antes do filtro de confiança).
  - **Total de objetos exibidos** (acima do limiar de confiança configurado).
  - **Contagem por classe** — ex.: `person: 3`, `car: 1`, `bicycle: 2` — ordenada por frequência.
  - **Detalhamento individual** de cada detecção filtrada, com: índice, rótulo, score percentual e coordenadas do bounding box (xmin, ymin, xmax, ymax).

### 4. Requisitos Não Funcionais (RNF) - Foco em Simplicidade
- **RNF01 - Sem Complexidade Adicional:** Não haverá sistemas de banco de dados, endpoints HTTP/API, sistemas de autenticação ou Dockerização elaborada.
- **RNF02 - Reproduzibilidade Prática:** O setup ocorrerá apenas através de um instalador e arquivo, sendo executável por meio de ambiente virtual (`.venv`) comum em desenvolvimento Python.
- **RNF03 - Aceleração com Hardware:** O ambiente deverá explicitamente utilizar os núcleos CUDA da placa de vídeo (GPU NVIDIA) disponíveis na máquina para reduzir o tempo de inferência ao carregar os pesos em `device="cuda"`.

### 5. Escopo Técnico e de Arquitetura Rápida
- **Linguagem Principal:** Python 3.9+
- **Bibliotecas-Chave Requeridas:** 
  - `transformers` (A base de comunicação da IA).
  - `torch` (PyTorch configurado com suporte nativo a CUDA para processamento através da GPU).
  - `Pillow` (Mapeamento inicial da estrutura da Imagem).
  - `matplotlib` (Opcional) ou próprio `Pillow` via ImageDraw para pintar as caixas nas saídas processadas.
- **Formato Recomendado de Execução:** 
  - Um notebook interativo (`.ipynb`), pois permite ao usuário rodar blocos isolados, facilitando a depuração e o entendimento do pipeline bloco a bloco sem recarregar o script inteiro. (Alternativamente, algo simples via `Gradio` se preferir UI visual).
  - **Módulo Sugerido Inicial:** `facebook/detr-resnet-50` (Tamanho tolerável e ótima precisão para o MVP).

### 6. Custo-Benefício do Esforço
- **Estimativa:** Construção integral e teste resolvidos em uma cadência rápida (algumas horas contínuas de laboratório/atividade).
- **Meta-Final:** Terminar o laboratório com pleno conhecimento em como carregar os pesos offline, inferir imagem e extrair a lógica e rótulos de forma programática utilizando IA aberta.

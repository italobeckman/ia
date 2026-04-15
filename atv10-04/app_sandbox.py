import gradio as gr
from transformers import pipeline
import torch
from PIL import Image, ImageDraw
from collections import Counter

def create_model_pipeline():
    # Detectar presença de placa de vídeo CUDA
    device = 0 if torch.cuda.is_available() else -1
    hw_name = "CUDA GPU" if device == 0 else "CPU (Lento)"
    print(f"[INFO] Carregando modelo no componente: {hw_name}")
    
    pipe = pipeline("object-detection", model="facebook/detr-resnet-50", device=device)
    return pipe

# Pipeline carregado em escopo global pra não re-instanciar toda requisição
global_pipe = None

THRESHOLD = 0.6  # Limiar de confiança mínima

def build_report(results_above_threshold, all_results, hw_name):
    """Gera o relatório técnico textual da inferência."""
    lines = []
    lines.append("=" * 48)
    lines.append("  RELATORIO TECNICO DE INFERENCIA")
    lines.append("=" * 48)
    lines.append(f"  Hardware utilizado : {hw_name}")
    lines.append(f"  Modelo             : facebook/detr-resnet-50")
    lines.append(f"  Limiar de confianca: {int(THRESHOLD * 100)}%")
    lines.append(f"  Total detectado    : {len(all_results)} objeto(s) bruto(s)")
    lines.append(f"  Acima do limiar    : {len(results_above_threshold)} objeto(s)")
    lines.append("-" * 48)

    if not results_above_threshold:
        lines.append("  Nenhum objeto detectado acima do limiar.")
    else:
        # Contagem por classe
        counter = Counter(r['label'] for r in results_above_threshold)
        lines.append("  CONTAGEM POR CLASSE:")
        for label, count in sorted(counter.items(), key=lambda x: -x[1]):
            plural = "objeto" if count == 1 else "objetos"
            lines.append(f"    - {label:<20} {count:>2} {plural}")
        lines.append("-" * 48)

        # Detalhamento individual ordenado por score
        lines.append("  DETALHAMENTO (ordenado por confianca):")
        sorted_results = sorted(results_above_threshold, key=lambda x: -x['score'])
        for i, r in enumerate(sorted_results, 1):
            b = r['box']
            lines.append(
                f"    #{i:02d} {r['label']:<18} {r['score']*100:.1f}%  "
                f"bbox=[{b['xmin']},{b['ymin']},{b['xmax']},{b['ymax']}]"
            )

    lines.append("=" * 48)
    return "\n".join(lines)

def detect_objects(image):
    global global_pipe
    if image is None:
        return None, "Envie uma imagem para iniciar a deteccao."

    if global_pipe is None:
        global_pipe = create_model_pipeline()

    hw_name = "CUDA GPU" if torch.cuda.is_available() else "CPU"

    # Inferência no modelo
    all_results = global_pipe(image)

    # Filtrar pelo limiar
    results_above = [r for r in all_results if r['score'] >= THRESHOLD]

    # Desenhando resultados na imagem
    draw = ImageDraw.Draw(image)
    for result in results_above:
        box = result['box']
        label = result['label']
        score = result['score']

        texto = f"{label.upper()} {score*100:.0f}%"
        coords = [(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])]
        draw.rectangle(coords, outline="red", width=4)
        draw.text((box['xmin'], max(0, box['ymin'] - 12)), texto, fill="red")

    # Gerar relatório textual
    report = build_report(results_above, all_results, hw_name)
    print(report)  # também loga no terminal

    return image, report


interface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil", label="Faca upload da Imagem de Teste"),
    outputs=[
        gr.Image(type="pil", label="Visualizacao do Resultado do Modelo"),
        gr.Textbox(
            label="Resultado Tecnico da Inferencia",
            lines=20,
        ),
    ],
    title="HF Sandbox - Object Detection",
    description=(
        "Deteccao de objetos via DETR (facebook/detr-resnet-50) com Hugging Face. "
        "O painel direito exibe o relatorio tecnico com contagem por classe e score de cada deteccao."
    ),
)

if __name__ == "__main__":
    interface.launch(inbrowser=True, server_port=7860)

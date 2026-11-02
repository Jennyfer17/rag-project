"""
Pipeline de ingestão de dados para o sistema RAG do Amigo do Agricultor.

Etapas:
  1. Obtém o texto de cada PDF de data/agriculture/:
     - Reutiliza o .txt já extraído em notebooks/data/extracted/, se existir
     - Caso contrário (ou com --reextract), extrai com PyMuPDF
     - Se o PDF for escaneado (sem texto), usa OCR via easyocr
  2. Limpa o texto e divide-o em chunks que cabem no modelo de embeddings
  3. Gera embeddings com sentence-transformers (modelo multilingue)
  4. Armazena na colecção configurada em RetrieverConfig, a mesma que a API lê

Uso (a partir de qualquer pasta):
    python src/ingest.py
    python src/ingest.py --reextract

Dependência adicional para OCR:
  pip install easyocr
"""

import argparse
import sys
from pathlib import Path

# Garante UTF-8 no stdout — necessário no Windows para barras de progresso
# do easyocr que usam caracteres fora do cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer

from rag.chunking import chunk_text, clean_text
from rag.config import RetrieverConfig

_CONFIG = RetrieverConfig()
PDF_DIR = _CONFIG.pdf_dir
TXT_DIR = _CONFIG.extracted_text_dir
CHROMA_DIR = _CONFIG.chroma_path
COLLECTION_NAME = _CONFIG.collection_name
EMBEDDING_MODEL = _CONFIG.embedding_model

# Mínimo de caracteres para considerar que o fitz extraiu texto útil
MIN_TEXT_LENGTH = 50

_ocr_reader = None


def _get_ocr_reader():
    """Carrega o leitor OCR apenas quando necessário (lazy loading)."""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            print("  [OCR] Carregando modelo easyocr (primeira vez pode demorar)...")
            _ocr_reader = easyocr.Reader(["pt", "en"], gpu=False)
        except ImportError:
            raise ImportError(
                "easyocr não está instalado. Execute: pip install easyocr"
            )
    return _ocr_reader


def _extract_with_ocr(pdf_path: Path) -> str:
    """Renderiza cada página como imagem e extrai texto via OCR."""
    reader = _get_ocr_reader()
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        # Renderiza a página em alta resolução (2x zoom → ~150 DPI efectivo)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")
        results = reader.readtext(img_bytes, detail=0, paragraph=True)
        page_text = "\n".join(results).strip()
        if page_text:
            pages.append(page_text)
        print(f"  [OCR] Página {i + 1}/{len(doc)} processada")
    doc.close()
    return "\n\n".join(pages)


def extract_pdf_to_txt(pdf_path: Path) -> str:
    """Extrai texto do PDF. Usa OCR como fallback se o PDF for escaneado."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    doc.close()
    text = "\n\n".join(pages)

    if len(text.strip()) < MIN_TEXT_LENGTH:
        print(f"  Texto insuficiente ({len(text.strip())} chars) — activando OCR...")
        text = _extract_with_ocr(pdf_path)

    return text


def load_source_text(pdf_path: Path, reextract: bool) -> str:
    """Devolve o texto do PDF, reutilizando o .txt existente salvo pedido em contrário."""
    txt_path = TXT_DIR / (pdf_path.stem + ".txt")
    if txt_path.exists() and not reextract:
        print(f"  A reutilizar texto extraído: {txt_path.name}")
        return txt_path.read_text(encoding="utf-8")

    text = extract_pdf_to_txt(pdf_path)
    txt_path.write_text(text, encoding="utf-8")
    print(f"  Salvo em: {txt_path}")
    return text


def main():
    parser = argparse.ArgumentParser(description="Reconstrói o índice dos manuais.")
    parser.add_argument(
        "--reextract",
        action="store_true",
        help="Extrai de novo o texto dos PDFs em vez de reutilizar os .txt existentes.",
    )
    args = parser.parse_args()

    TXT_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"Nenhum PDF encontrado em {PDF_DIR}")
        sys.exit(1)

    print(f"Encontrados {len(pdf_files)} PDFs\n")

    all_chunks: list[str] = []
    all_metadatas: list[dict] = []
    all_ids: list[str] = []

    print(f"A carregar o modelo de embeddings: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    def count_tokens(text: str) -> int:
        return len(model.tokenizer.encode(text, add_special_tokens=True, truncation=False))

    # --- Etapa 1 & 2: Extração, limpeza e chunking ---
    for pdf_path in pdf_files:
        print(f"Processando: {pdf_path.name}")
        try:
            text = clean_text(load_source_text(pdf_path, args.reextract))
        except Exception as e:
            print(f"  ERRO ao extrair {pdf_path.name}: {e}")
            continue

        chunks = chunk_text(
            text,
            max_size=_CONFIG.chunk_max_tokens,
            overlap=_CONFIG.chunk_overlap_tokens,
            count_tokens=count_tokens,
        )
        print(f"  Chunks gerados: {len(chunks)}")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": pdf_path.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })
            all_ids.append(f"{pdf_path.stem}_chunk_{i}")

    print(f"\nTotal de chunks: {len(all_chunks)}")

    truncated = sum(count_tokens(chunk) > model.max_seq_length for chunk in all_chunks)
    print(f"Chunks acima do limite do modelo ({model.max_seq_length} tokens): {truncated}")

    # --- Etapa 3: Embeddings ---
    print("Gerando embeddings...")
    embeddings = model.encode(
        all_chunks,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True,
    )

    # --- Etapa 4: Armazenamento no ChromaDB ---
    print("\nArmazenando no ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Coleção '{COLLECTION_NAME}' existente removida.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch_end = min(i + batch_size, len(all_chunks))
        collection.add(
            documents=all_chunks[i:batch_end],
            embeddings=embeddings[i:batch_end].tolist(),
            metadatas=all_metadatas[i:batch_end],
            ids=all_ids[i:batch_end],
        )
        print(f"  Armazenados chunks {i + 1}–{batch_end} / {len(all_chunks)}")

    total = collection.count()
    print(f"\nConcluído! Coleção '{COLLECTION_NAME}' contém {total} documentos.")
    print(f"Base vectorial salva em: {CHROMA_DIR.resolve()}")


if __name__ == "__main__":
    main()

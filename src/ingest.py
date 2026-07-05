"""
Pipeline de ingestão de dados para o sistema RAG do Amigo do Agricultor.

Etapas:
  1. Extrai texto dos PDFs em data/agriculture/ → salva em data/extracted/
     - Tenta extracção directa com PyMuPDF
     - Se o PDF for escaneado (sem texto), usa OCR via easyocr
  2. Divide o texto em chunks com overlap
  3. Gera embeddings com sentence-transformers (modelo multilingual)
  4. Armazena na base vectorial ChromaDB em data/chroma_db/

Dependência adicional para OCR:
  pip install easyocr
"""

import re
import sys
from pathlib import Path

# Garante UTF-8 no stdout — necessário no Windows para barras de progresso
# do easyocr que usam caracteres fora do cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer


PDF_DIR = Path("data/agriculture")
TXT_DIR = Path("data/extracted")
CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "agriculture_manuals"

CHUNK_SIZE = 1000       # caracteres por chunk
CHUNK_OVERLAP = 200     # sobreposição entre chunks
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

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


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\x00", "", text)
    return text.strip()


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]

        if end < len(text):
            boundary = max(
                chunk.rfind(". "),
                chunk.rfind(".\n"),
                chunk.rfind("! "),
                chunk.rfind("? "),
                chunk.rfind("\n\n"),
            )
            if boundary > CHUNK_SIZE // 2:
                end = start + boundary + 1
                chunk = text[start:end]

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP

    return chunks


def main():
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

    # --- Etapa 1 & 2: Extração e chunking ---
    for pdf_path in pdf_files:
        print(f"Extraindo: {pdf_path.name}")
        try:
            raw_text = extract_pdf_to_txt(pdf_path)
            text = clean_text(raw_text)
        except Exception as e:
            print(f"  ERRO ao extrair {pdf_path.name}: {e}")
            continue

        txt_path = TXT_DIR / (pdf_path.stem + ".txt")
        txt_path.write_text(text, encoding="utf-8")
        print(f"  Salvo em: {txt_path}")

        chunks = chunk_text(text)
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

    # --- Etapa 3: Embeddings ---
    print(f"\nCarregando modelo de embeddings: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

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

#data\extracted\5.-Rec.-Agr.-Maputo.txt
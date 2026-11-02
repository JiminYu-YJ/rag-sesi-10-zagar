import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def jalankan_lab2():
    print("--- 🧪 LAB 2: Membangun Indeks & Pencarian Semantik (Multi-PDF) ---")
    
    folder_dokumen = "dokumen"
    teks_gabungan = ""
    
    if not os.path.exists(folder_dokumen):
        print(f"❌ Folder '{folder_dokumen}' tidak ditemukan!")
        return

    # Mendeteksi otomatis semua file PDF di dalam folder dokumen/
    daftar_pdf = [f for f in os.listdir(folder_dokumen) if f.lower().endswith('.pdf')]
    
    if not daftar_pdf:
        print(f"❌ Tidak ada file PDF di dalam folder '{folder_dokumen}/'!")
        return

    print(f"📂 File PDF yang ditemukan: {daftar_pdf}")

    for nama_file in daftar_pdf:
        file_path = os.path.join(folder_dokumen, nama_file)
        reader = PdfReader(file_path)
        teks_file = ""
        for halaman in reader.pages:
            t = halaman.extract_text()
            if t:
                teks_file += t + "\n"
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
        chunks_file = splitter.split_text(teks_file)
        print(f"   - {nama_file}: menghasilkan {len(chunks_file)} chunk.")
        teks_gabungan += teks_file + "\n"

    # Chunking keseluruhan teks dari semua PDF
    splitter_utama = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
    semua_chunks = splitter_utama.split_text(teks_gabungan)
    print(f"\n📄 Total keseluruhan chunk dari semua PDF: {len(semua_chunks)}")

    # Inisialisasi ChromaDB & Embedding Lokal
    client = chromadb.Client()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    collection = client.get_or_create_collection(name="rag_lab2_multi", embedding_function=ef)

    # Masukkan chunk ke Vector Database
    collection.add(
        documents=semua_chunks,
        ids=[str(i) for i in range(len(semua_chunks))]
    )
    print("✅ Berhasil menyimpan seluruh indeks ke Vector Database (ChromaDB)!\n")

    # Mengajukan 5 Pertanyaan (Sesuai ketentuan Lab 2)
    daftar_pertanyaan = [
        "Apa itu Retrieval-Augmented Generation?",
        "Mengapa chunking diperlukan dalam sistem RAG?",
        "Bagaimana cara kerja vector database?",
        "Jelaskan teknik memecah dokumen agar tidak kehilangan konteks!",
        "Bagaimana cara melakukan optimasi sistem RAG?"
    ]

    print("--- 🔍 HASIL PENCARIAN QUERY (5 PERTANYAAN) ---")
    for i, q in enumerate(daftar_pertanyaan, 1):
        hasil = collection.query(query_texts=[q], n_results=1)
        print(f"\nPertanyaan {i}: {q}")
        print(f"💡 Hasil Chunk Relevan: {hasil['documents'][0][0].strip()}")
        print("-" * 60)

if __name__ == "__main__":
    jalankan_lab2()
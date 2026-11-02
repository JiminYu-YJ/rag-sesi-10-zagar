import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def jalankan_lab3():
    print("--- 🧪 LAB 3: RAG Pipeline & Evaluasi Pencarian ---")
    
    folder_dokumen = "dokumen"
    teks_gabungan = ""
    
    if not os.path.exists(folder_dokumen):
        print(f"❌ Folder '{folder_dokumen}' tidak ditemukan!")
        return

    # 1. Otomatis membaca semua file PDF di folder dokumen/
    daftar_pdf = [f for f in os.listdir(folder_dokumen) if f.lower().endswith('.pdf')]
    print(f"📂 Memuat dokumen: {daftar_pdf}")

    for nama_file in daftar_pdf:
        file_path = os.path.join(folder_dokumen, nama_file)
        reader = PdfReader(file_path)
        for halaman in reader.pages:
            t = halaman.extract_text()
            if t:
                teks_gabungan += t + "\n"

    # 2. Proses Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
    semua_chunks = splitter.split_text(teks_gabungan)
    print(f"📄 Total chunk yang diindeks: {len(semua_chunks)}")

    # 3. Inisialisasi Vector Database (ChromaDB) & Embedding Lokal
    client = chromadb.Client()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    # Gunakan collection unik untuk Lab 3
    collection = client.get_or_create_collection(name="rag_lab3_pipeline", embedding_function=ef)
    
    collection.add(
        documents=semua_chunks,
        ids=[str(i) for i in range(len(semua_chunks))]
    )
    print("✅ Berhasil membangun indeks untuk RAG Pipeline!\n")

    # 4. Simulasi RAG (Retrieval + Context Generation)
    pertanyaan_pengguna = "Bagaimana cara kerja vector database dan apa fungsinya?"
    
    print(f"❓ Pertanyaan Pengguna: \"{pertanyaan_pengguna}\"")
    
    # Melakukan retrieval (pencarian chunk relevan teratas)
    hasil_retrieval = collection.query(query_texts=[pertanyaan_pengguna], n_results=2)
    konteks_terambil = hasil_retrieval['documents'][0]

    print("\n🔍 Konteks Relevan yang Ditemukan dari Dokumen:")
    for idx, teks_chunk in enumerate(konteks_terambil, 1):
        print(f"   [Chunk {idx}]: {teks_chunk.strip()}")
        print("-" * 40)

    # 5. Simulasi Jawaban Akhir Berdasarkan Konteks (RAG Generation)
    print("\n🤖 Simulasi Jawaban Sistem RAG:")
    print("Berdasarkan dokumen yang tersedia, vector database bekerja dengan menyimpan representasi numerik (embedding) dari teks ke dalam ruang multi-dimensi. Ketika ada pertanyaan, sistem mencari jarak terdekat (cosine similarity) untuk mencocokkan informasi yang paling relevan guna menjawab kebutuhan pengguna.")

if __name__ == "__main__":
    jalankan_lab3()
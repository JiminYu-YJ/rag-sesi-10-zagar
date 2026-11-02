import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def jalankan_lab1():
    print("--- 🧪 LAB 1: Eksperimen Chunking (Multi-PDF) ---")
    
    folder_dokumen = "dokumen"
    if not os.path.exists(folder_dokumen):
        print(f"❌ Folder '{folder_dokumen}' tidak ditemukan!")
        return

    # Mendeteksi otomatis semua file PDF di dalam folder dokumen/
    daftar_pdf = [f for f in os.listdir(folder_dokumen) if f.lower().endswith('.pdf')]
    
    if not daftar_pdf:
        print(f"❌ Tidak ada file PDF di dalam folder '{folder_dokumen}/'!")
        return

    print(f"📂 File PDF yang ditemukan: {daftar_pdf}")
    
    teks_lengkap = ""
    for nama_file in daftar_pdf:
        file_path = os.path.join(folder_dokumen, nama_file)
        reader = PdfReader(file_path)
        for halaman in reader.pages:
            teks = halaman.extract_text()
            if teks:
                teks_lengkap += teks + "\n"
            
    print(f"📄 Berhasil membaca seluruh dokumen. Total panjang teks: {len(teks_lengkap)} karakter.")
    print("-" * 50)

    # Menguji tiga ukuran chunk (500, 1000, 1500) dengan overlap 15%
    ukuran_chunk_list = [500, 1000, 1500]

    for ukuran in ukuran_chunk_list:
        overlap = int(ukuran * 0.15) # Overlap 15%
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=ukuran,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = splitter.split_text(teks_lengkap)
        
        print(f"Ukuran Chunk: {ukuran} | Overlap: {overlap} | Total Chunk Dihasilkan: {len(chunks)}")
        if chunks:
            print(f"Contoh chunk pertama:\n\"{chunks[0][:150]}...\"\n")
        print("=" * 50)

if __name__ == "__main__":
    jalankan_lab1()
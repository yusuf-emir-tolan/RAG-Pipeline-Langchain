# RAG Pipeline — Web Document Q&A

LangChain + ChromaDB + HuggingFace + Groq ile web dökümanı üzerinde soru-cevap sistemi.

## Nasıl Çalışır

Web sayfası yüklenir → Metin parçalara bölünür → HuggingFace ile embedding oluşturulur → ChromaDB'ye kaydedilir → Kullanıcı sorusu gelir → İlgili chunk bulunur → Groq LLM cevap üretir (streaming)

## Teknolojiler

LangChain — RAG pipeline
ChromaDB — Vektör veritabanı
HuggingFace — Embedding (all-MiniLM-L6-v2)
Groq — LLM (Qwen3.6-27b)
RecursiveCharacterTextSplitter — chunk_size=1000, overlap=200

## Kurulum

Repoyu klonla: git clone https://github.com/yusuf-emir-tolan/rag-pipeline-langchain
Bağımlılıkları yükle: pip install -r requirements.txt
.env.example dosyasını .env olarak kopyala
GROQ_API_KEY'ini .env dosyasına ekle
Çalıştır: python main.py

## Not

hub.pull() deprecated olduğu için ChatPromptTemplate.from_template() kullanıldı.

## Yazar

Yusuf Emir Tolan
LinkedIn | GitHub

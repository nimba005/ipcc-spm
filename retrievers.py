import faiss
import numpy as np
import pickle
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class FAISSRetriever(BaseRetriever):
    def __init__(self, report_key, index_path, texts_path, embed_model, k=5):
        super().__init__()
        self._report_key = report_key
        self._embed_model = embed_model
        self._index = faiss.read_index(index_path)
        with open(texts_path, 'rb') as f:
            self._texts = pickle.load(f)
        self._k = k

    def _get_relevant_documents(self, query: str):
        query_embedding = self._embed_model.embed_query(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        distances, indices = self._index.search(query_embedding, self._k * 20)
        relevant_docs = []
        for i, distance in zip(indices[0], distances[0]):
            if len(relevant_docs) >= self._k:
                break
            if distance > 0.65:
                continue
            text_entry = self._texts[i]
            text = text_entry['text'] if isinstance(text_entry, dict) else str(text_entry)
            metadata = text_entry.get('metadata', {}) if isinstance(text_entry, dict) else {}
            report_key = metadata.get('report_key', 'Unknown')
            if report_key == self._report_key:
                relevant_docs.append(Document(page_content=text, metadata=metadata))
        return relevant_docs

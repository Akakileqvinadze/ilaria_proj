from pathlib import Path
from typing import List, Dict
import re


class DocumentLoader:
    def __init__(self, data_path: str = "app/data"):
        self.data_path = Path(data_path)
        self.file_path = self.data_path / "ilaria-instructions.md"

    def load_document(self) -> str:
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"❌ ვერ მოიძებნა: {self.file_path}\n"
                "გთხოვთ მოათავსოთ ilaria-instructions.md ფაილი app/data/ დირექტორიაში"
            )

        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"✅ ჩაიტვირთა დოკუმენტი: {len(content)} სიმბოლო")
        return content

    def split_into_chunks(
        self, 
        text: str, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> List[Dict]:
        sections = re.split(r'\n(?=#+\s)', text)
        chunks = []
        chunk_id = 0

        for section in sections:
            title_match = re.match(r'#+\s+(.+)', section)
            section_title = title_match.group(1) if title_match else "Unknown"

            section_chunks = self._split_text_by_size(
                section, chunk_size, chunk_overlap
            )

            for chunk_text in section_chunks:
                chunks.append({
                    "id": chunk_id,
                    "content": chunk_text,
                    "metadata": {
                        "source": "ilaria-instructions.md",
                        "section": section_title,
                        "chunk_id": chunk_id
                    }
                })
                chunk_id += 1

        print(f"✅ დაიყო {len(chunks)} ნაწილად")
        return chunks

    def _split_text_by_size(
        self, text: str, chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            if end < len(text):
                for sep in ['\n\n', '\n', '. ', '! ', '? ']:
                    pos = text.rfind(sep, start, end)
                    if pos != -1:
                        end = pos + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - chunk_overlap

        return chunks

    def load_and_split(
        self, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[Dict]:
        content = self.load_document()
        chunks = self.split_into_chunks(content, chunk_size, chunk_overlap)
        return chunks

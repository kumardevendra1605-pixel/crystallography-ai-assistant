from src.docx_loader import load_docx

docs = load_docx("data/docx")

print("Documents loaded:", len(docs))
print(docs[0]["content"][:500])
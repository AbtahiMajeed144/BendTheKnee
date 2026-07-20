import os
import PyPDF2

pdf_files = [
    "2403.13117v3.pdf",
    "2509.24936v2.pdf",
    "2510.26645v1.pdf",
    "2512.16768v3.pdf",
    "2604.04491v1.pdf"
]

with open("summary.txt", "w", encoding="utf-8") as out:
    for pdf in pdf_files:
        try:
            reader = PyPDF2.PdfReader(pdf)
            out.write(f"\n\n{'='*50}\nPAPER: {pdf}\n{'='*50}\n")
            # Read first 3 pages
            num_pages = min(3, len(reader.pages))
            for i in range(num_pages):
                page = reader.pages[i]
                text = page.extract_text()
                if text:
                    out.write(f"\n--- Page {i+1} ---\n")
                    out.write(text)
        except Exception as e:
            out.write(f"Error reading {pdf}: {e}\n")

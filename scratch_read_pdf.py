import PyPDF2
import glob

with open('scratch_pdf_output.txt', 'w', encoding='utf-8') as out_f:
    for f in glob.glob('curl-certificate/literatures/*.pdf'):
        out_f.write(f'\n--- {f} --- \n')
        try:
            reader = PyPDF2.PdfReader(f)
            text = ''
            for i in range(min(5, len(reader.pages))):
                text += reader.pages[i].extract_text() + '\n'
            out_f.write(text[:2000] + '\n')
        except Exception as e:
            out_f.write(f"Error: {e}\n")

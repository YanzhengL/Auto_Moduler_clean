import PyPDF2


def pdf_to_text(pdf_file_path, txt_file_path):
    # Open the PDF file
    with open(pdf_file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""

        # Extract text from each page
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    # Save the extracted text to a .txt file
    with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)


# Example usage
pdf_file = "C:\\Users\\psxyl37\\Documents\\papers\\2. Digital_Twins_for_Modern_Power_Electronics_An_investigation_into_the_enabling_technologies_and_applications.pdf"  # Replace with your PDF file path
txt_file = "../data/digital_twin_paper.txt"  # Replace with your desired output file path
pdf_to_text(pdf_file, txt_file)

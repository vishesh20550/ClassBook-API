from flask import Flask, request, jsonify
from final import process_pdf_from_url

app = Flask(__name__)

@app.route('/', methods=['POST'])
def convert_pdf():
    data = request.get_json()
    pdf_id = data['pdfId']

    # Call your model function to convert PDF to text
    extracted_text = process_pdf_from_url(pdf_id)

    return jsonify({'script': extracted_text})


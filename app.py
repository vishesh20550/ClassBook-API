from flask import Flask, request, jsonify
from final import process_pdf_from_id

app = Flask(__name__)

@app.route('/', methods=['POST'])
def convert_pdf():
    data = request.get_json()
    pdf_id = data['pdfId']

    # Call your model function to convert PDF to text
    script_id = process_pdf_from_id(pdf_id)

    return jsonify({'script_id': script_id})


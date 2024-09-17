from flask import Request, jsonify
from final import process_pdf_from_id

def convert_pdf(request: Request):
    # Parse the JSON request
    data = request.get_json(silent=True)
    if data is None or 'pdfId' not in data:
        return jsonify({'error': 'Invalid request, "pdfId" not provided'}), 400

    pdf_id = data['pdfId']

    # Call your function to process the PDF
    try:
        script_id = process_pdf_from_id(pdf_id)
    except Exception as e:
        # Handle exceptions and return an error response
        return jsonify({'error': str(e)}), 500

    # Return the script ID as a JSON response
    return jsonify({'script_id': script_id}), 200

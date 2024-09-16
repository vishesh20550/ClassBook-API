import os
import json
import base64
from uuid import uuid4
from google.cloud import vision_v1
from google.oauth2 import service_account
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, storage as firebase_storage

def process_pdf_from_id(pdf_id):
    """
    Retrieves a PDF from Firebase Storage using pdf_id, processes it through OCR and text restructuring,
    generates a teleprompting script, saves it in Firebase Storage, and returns the script_id.
    """
    # ---------------------- Configuration ----------------------
    # Load API keys and credentials from environment variables
    GEMINI_API_KEY = os.environ.get('GEMINI_KEY')
    GOOGLE_SERVICE_BASE64 = os.environ.get('GOOGLE_SERVICE_BASE64')
    # FIREBASE_CREDENTIALS_BASE64 = os.environ.get('FIREBASE_CREDENTIALS_BASE64')
    FIREBASE_STORAGE_BUCKET = os.environ.get('FIREBASE_STORAGE_BUCKET')  # e.g., 'your-app.appspot.com'

    if not GEMINI_API_KEY:
        raise EnvironmentError('GEMINI_KEY environment variable not found')

    if not GOOGLE_SERVICE_BASE64:
        raise EnvironmentError('GOOGLE_SERVICE_BASE64 environment variable not found')

    # if not FIREBASE_CREDENTIALS_BASE64:
    #     raise EnvironmentError('FIREBASE_CREDENTIALS_BASE64 environment variable not found')

    if not FIREBASE_STORAGE_BUCKET:
        raise EnvironmentError('FIREBASE_STORAGE_BUCKET environment variable not found')

    # Load Google Cloud credentials
    google_service_content = base64.b64decode(GOOGLE_SERVICE_BASE64)
    service_account_info = json.loads(google_service_content)
    credentials_gc = service_account.Credentials.from_service_account_info(service_account_info)

    # Initialize Firebase Admin SDK
    # firebase_creds_content = base64.b64decode(FIREBASE_CREDENTIALS_BASE64)
    # firebase_creds = json.loads(firebase_creds_content)
    cred = credentials.Certificate(service_account_info)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {'storageBucket': FIREBASE_STORAGE_BUCKET})

    # Initialize clients
    genai.configure(api_key=GEMINI_API_KEY)
    vision_client = vision_v1.ImageAnnotatorClient(credentials=credentials_gc)
    firebase_bucket = firebase_storage.bucket()

    # ---------------------- Retrieve PDF from Firebase Storage ----------------------
    # pdf_filename = f'class_pdfs/{pdf_id}'
    # pdf_blob = firebase_bucket.blob(pdf_filename)
    # if not pdf_blob.exists():
    #     print(f"PDF with ID {pdf_filename} does not exist in Firebase Storage.")
    #     return None

    # pdf_content = pdf_blob.download_as_bytes()
    # print(f"Retrieved PDF from Firebase Storage: {pdf_filename}")

    # # ---------------------- Upload PDF to Temporary Location in Firebase Storage ----------------------
    # temp_pdf_filename = f'temporary/{pdf_id}'
    # temp_pdf_blob = firebase_bucket.blob(temp_pdf_filename)
    # temp_pdf_blob.upload_from_string(pdf_content, content_type='application/pdf')
    # gcs_source_uri = f'gs://{FIREBASE_STORAGE_BUCKET}/{temp_pdf_filename}'
    gcs_source_uri = f'gs://{FIREBASE_STORAGE_BUCKET}/class_pdfs/{pdf_id}'
    

    # print(f"Uploaded PDF to temporary location: {gcs_source_uri}")

    # ---------------------- Perform OCR ----------------------
    def detect_text_from_pdf(gcs_source_uri):
        # Specify input configuration
        input_config = vision_v1.InputConfig(
            gcs_source=vision_v1.GcsSource(uri=gcs_source_uri),
            mime_type='application/pdf'
        )

        # Specify output configuration to Firebase Storage
        output_uri = f'gs://{FIREBASE_STORAGE_BUCKET}/temporary/vision_output/{pdf_id}/'
        output_config = vision_v1.OutputConfig(
            gcs_destination=vision_v1.GcsDestination(uri=output_uri),
            batch_size=2  # Process two pages at a time
        )

        # Create the request
        request = vision_v1.AsyncAnnotateFileRequest(
            features=[vision_v1.Feature(type_=vision_v1.Feature.Type.DOCUMENT_TEXT_DETECTION)],
            input_config=input_config,
            output_config=output_config
        )

        # Perform the asynchronous request
        operation = vision_client.async_batch_annotate_files(requests=[request])

        # Wait for the operation to complete
        print('Waiting for the OCR operation to complete...')
        response = operation.result(timeout=1000)

        # Return the output URI
        return output_uri

    # ---------------------- Download OCR Output ----------------------
    def download_ocr_output(output_uri):
        # The output is written to JSON files in Firebase Storage
        # We'll download and parse it directly
        prefix = output_uri.replace(f'gs://{FIREBASE_STORAGE_BUCKET}/', '')
        blobs = firebase_bucket.list_blobs(prefix=prefix)

        full_text = ''
        for blob in blobs:
            json_string = blob.download_as_string()
            response = vision_v1.AnnotateFileResponse.from_json(json_string)
            for annotation in response.responses:
                if annotation.full_text_annotation.text:
                    full_text += annotation.full_text_annotation.text

        print("OCR text downloaded.")
        return full_text

    # ---------------------- Process Text with Gemini ----------------------
    def process_text_with_gemini(text_to_process):
        # Prepare the prompt
        prompt = f"""
Please organize the following text by improving its structure, coherence, and readability.
Ensure the text remains true to the original content.
Each section should begin with its respective page number, and any references to textbook images should be preserved or enhanced for clarity.
Do not add any new information.

Text to process:
{text_to_process}
"""
        # Send the structured prompt to the Gemini API and get the response
        response = genai.generate_content(
            model="models/gemini-1.5-flash",
            prompt=prompt
        )

        print("Processed text with Gemini.")
        return response.result

    # ---------------------- Generate Teleprompting Script ----------------------
    def generate_teleprompting_script(text_to_process):
        prompt = f"""
Please generate a detailed teleprompting script for a teacher based on the following content.

This script should be designed for the teacher to read aloud in class, covering all key concepts comprehensively over seven periods, 
with each period lasting 35 minutes (with a 10-minute buffer for activities or distractions). 
The content should be enough to last for the full duration of each period.

The script must be **detailed and specific**, providing exact phrasing for the teacher to read, ensuring it lasts for the required time. 
Keep in mind that EACH period will be of 35 minutes, EACH period will have introduction and recap of 5 minutes, main lesson of 25 minutes, activity or discussion of 5 minutes and recap and conclusion of 5 minutes. The content should be detailed and engaging, with specific questions and prompts for the teacher to read aloud.
Please make sure the content is detailed enough to last for the full duration of each period.

It should follow this structure for each period:

### Period Structure:
1. **Introduction and Recap (5 minutes)**:
   - Include an exact recap of the previous period's content. Make sure period 1 has no recap since it is the first period.
   - Provide the text for the teacher to introduce the day's lesson and prepare students for new concepts.

2. **Main Lesson (25 minutes)**:
   - Provide extensive content for the teacher to read, explaining key concepts in detail.
   - For every conceptual term, please explain what it means like the teacher is explaining to the students and further explain the concept in detail, not in one-liners.
   - Ensure all important terms and concepts are explained thoroughly, using natural, engaging language.
   - Please add content worth reading for 25 minutes.
   - When introducing videos or diagrams, include **detailed instructions** for the teacher on what to say before and after the media.

3. **Activity or Discussion (5 minutes)**:
   - Suggest a detailed, **specific activity or discussion** related to the lesson.
   - Provide exact language for the teacher to introduce and facilitate the activity (e.g., *"Now, let's do a quick pair-and-share activity. Discuss with your partner how different beak shapes help birds eat certain types of food."*).
   - Either suggest some YouTube video related to the content.

4. **Recap and Conclusion (5 minutes)**:
   - Provide a detailed recap of the day's lesson.
   - Suggest a final reflective question or thought for students to think about (e.g., *"How do adaptations like feathers help birds survive in different environments?"*).
   - Offer specific next steps or questions for students to consider before the next period.

Ensure that the content flows naturally, is engaging, and clear. The script must be long enough to fill the required time for each section. Here is the content for the script:

{text_to_process}
"""
        # Send the prompt to the Gemini API and get the response
        response = genai.generate_content(
            model="models/gemini-1.5-flash",
            prompt=prompt
        )

        print("Generated teleprompting script.")
        return response.result

    # ---------------------- Main Processing ----------------------
    try:
        # Perform OCR
        output_uri = detect_text_from_pdf(gcs_source_uri)

        # Download OCR output
        ocr_text = download_ocr_output(output_uri)

        # Process text with Gemini
        processed_text = process_text_with_gemini(ocr_text)

        # Generate the final script
        final_script = generate_teleprompting_script(processed_text)

        # Generate a unique script ID
        script_id = str(uuid4())

        # Upload the final script to Firebase Storage
        script_filename = f'class_scripts/{script_id}.txt'
        script_blob = firebase_bucket.blob(script_filename)
        script_blob.upload_from_string(final_script, content_type='text/plain')

        print(f"Final script uploaded to Firebase Storage: {script_filename}")

        return script_id  # Return the script ID

    finally:
        # Clean up temporary files in Firebase Storage
        print("Cleaning up temporary files in Firebase Storage.")
        # Delete the temporary uploaded PDF
        # temp_pdf_blob.delete()
        # Delete OCR output files
        prefix = f'temporary/vision_output/{pdf_id}/'
        blobs = firebase_bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            blob.delete()
        print("Cleanup complete.")

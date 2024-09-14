import google.generativeai as genai
from google.oauth2 import service_account
from google.cloud import vision_v1
from google.cloud import storage
import json
import requests
import os


def process_pdf_from_url(pdf_url):
    """
    Downloads a PDF from a URL and processes it through OCR and text restructuring,
    then generates a teleprompting script.

    Args:
        pdf_url (str): The URL to the PDF file.
    """
    # ---------------------- Configuration ----------------------
    # Set your API keys and credentials securely
    # Replace with your actual API key
    with open('config.json', 'r') as file:
        data = json.load(file)
    GEMINI_API_KEY = data["gemini_key"]

    # Set the environment variable to the path of your Google Cloud service account key
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'secret-helper-435208-s0-683d7868b041.json'

    # Google Cloud storage bucket details
    bucket_name = 'bucketone__one'  
    output_prefix = 'vision_output'

    # Initialize clients
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    vision_client = vision_v1.ImageAnnotatorClient()
    storage_client = storage.Client()

    # Create the Output directory if it doesn't exist
    if not os.path.exists('Output'):
        os.makedirs('Output')

    # ---------------------- Download PDF ----------------------
    pdf_file_path = 'temp_downloaded.pdf'
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(pdf_file_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded PDF from {pdf_url}")
    else:
        print(f"Failed to download PDF from {pdf_url}")
        return

    # ---------------------- Upload PDF to GCS ----------------------
    def upload_pdf_to_gcs(bucket_name, pdf_file_path):
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(os.path.basename(pdf_file_path))

        # Upload the PDF to the specified bucket
        blob.upload_from_filename(pdf_file_path)
        print(f"File {pdf_file_path} uploaded to {bucket_name}.")

    # ---------------------- Perform OCR ----------------------
    def detect_text_from_pdf(bucket_name, output_prefix, gcs_source_uri):
        # Specify input configuration
        input_config = vision_v1.InputConfig(
            gcs_source=vision_v1.GcsSource(uri=gcs_source_uri),
            mime_type='application/pdf'
        )

        # Specify output configuration
        gcs_destination_uri = f'gs://{bucket_name}/{output_prefix}/'
        output_config = vision_v1.OutputConfig(
            gcs_destination=vision_v1.GcsDestination(uri=gcs_destination_uri),
            batch_size=2  # Process two pages at a time
        )

        # Create the request
        request = vision_v1.AsyncAnnotateFileRequest(
            features=[vision_v1.Feature(
                type_=vision_v1.Feature.Type.DOCUMENT_TEXT_DETECTION)],
            input_config=input_config,
            output_config=output_config
        )

        # Perform the asynchronous request
        operation = vision_client.async_batch_annotate_files(requests=[
                                                             request])

        # Wait for the operation to complete
        print('Waiting for the operation to complete...')
        response = operation.result(timeout=180)

        # Get the output from GCS
        return gcs_destination_uri

    # ---------------------- Download OCR Output ----------------------
    def download_ocr_output(output_uri, destination_file):
        bucket_name = output_uri.split('/')[2]
        prefix = '/'.join(output_uri.split('/')[3:])

        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)

        with open(destination_file, 'w',encoding="utf-8") as output_file:
            for blob in blobs:
                json_string = blob.download_as_string()
                response = vision_v1.AnnotateFileResponse.from_json(
                    json_string)

                for annotation in response.responses:
                    if annotation.full_text_annotation.text:
                        output_file.write(annotation.full_text_annotation.text)

        print(f"OCR text saved to {destination_file}")

    # ---------------------- Process PDF to Text ----------------------
    def process_pdf_to_text(pdf_file_path, bucket_name, output_prefix):
        # Upload the PDF to GCS
        upload_pdf_to_gcs(bucket_name, pdf_file_path)

        # Construct the GCS URI
        gcs_source_uri = f'gs://{bucket_name}/{os.path.basename(pdf_file_path)}'

        # Perform OCR and get the output URI
        gcs_output_uri = detect_text_from_pdf(
            bucket_name, output_prefix, gcs_source_uri)

        # Download and save the OCR result
        output_text_file = 'Output/scriptGen.txt'
        download_ocr_output(gcs_output_uri, output_text_file)

    # ---------------------- Process Text with Gemini ----------------------
    def process_text_with_gemini(input_text_file):
        # Read the text from the file
        with open(input_text_file, 'r',encoding="utf-8") as file:
            text_to_process = file.read()

        # Prompt for the Gemini model
        prompt = f"""
Please organize the following text by improving its structure, coherence, and readability.
Ensure the text remains true to the original content.
Each section should begin with its respective page number, and any references to textbook images should be preserved or enhanced for clarity.
Do not add any new information.

Text to process:
{text_to_process}
"""

        # Send the structured prompt to the Gemini API and get the response
        response = model.generate_content(prompt)

        # Path for the output text file
        output_text_file = 'Output/scriptGenFinal.txt'

        # Update the text file with the restructured and cited text
        with open(output_text_file, 'w',encoding="utf-8") as file:
            file.write(response.text)

        print(
            f"File '{output_text_file}' has been updated with restructured and cited text.")
        return output_text_file

    # ---------------------- Generate Teleprompting Script ----------------------
    def generate_teleprompting_script(input_text_file):
        output_text_file = 'Output/finalScript.txt'

        with open(input_text_file, 'r',encoding="utf-8") as file:
            text_to_process = file.read()

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
   - For every conceptual term please explain what it means like the teacher is explaining to the students and further explain the concept in detail not in one liners.
   - Ensure all important terms and concepts are explained thoroughly, using natural, engaging language.
   - Please add content worth reading for 25 minutes
   - When introducing videos or diagrams, include **detailed instructions** for the teacher on what to say before and after the media.

3. **Activity or Discussion (5 minutes)**:
   - Suggest a detailed, **specific activity or discussion** related to the lesson.
   - Provide exact language for the teacher to introduce and facilitate the activity (e.g., *"Now, let's do a quick pair-and-share activity. Discuss with your partner how different beak shapes help birds eat certain types of food."*).
   - Either suggest some YouTube video related to the content.

4. **Recap and Conclusion (5 minutes)**:
   - Provide a detailed recap of the day's lesson.
   - Suggest a final reflective question or thought for students to think about (e.g., *"How do adaptations like feathers help birds survive in different environments?"*).
   - Offer specific next steps or questions for students to consider before the next period.

Ensure that the content flows naturally, and is engaging and clear. The script must be long enough to fill the required time for each section. Here is the content for the script:

{text_to_process}
"""

        # Send the prompt to the Gemini API and get the response
        response = model.generate_content(prompt)

        # Save the generated script to a file
        with open(output_text_file, 'w') as file:
            file.write(response.text)

        print(f"Class script generated and saved to {output_text_file}.")
        return response.text
    # ---------------------- Main Processing ----------------------
    try:
        # Process the PDF to extract text
        process_pdf_to_text(pdf_file_path, bucket_name, output_prefix)

        # Process the extracted text with Gemini
        final_text_file = process_text_with_gemini('Output/scriptGen.txt')

        # Generate the teleprompting script
        return generate_teleprompting_script(final_text_file)
    finally:
        # Clean up the temporary PDF file
        if os.path.exists(pdf_file_path):
            os.remove(pdf_file_path)
            print(f"Removed temporary file {pdf_file_path}")

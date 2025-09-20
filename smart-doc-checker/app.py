# This script requires Streamlit, NLTK, NumPy, and scikit-learn.
# Install them with: pip install -r requirements.txt
import streamlit as st
import os
import datetime
import zipfile
import io
import heapq

# Attempt to import scientific computing libraries and provide a friendly error if they are missing.
# This check ensures the app doesn't crash and guides the user to install dependencies.
try:
    from fpdf import FPDF
    from docx import Document
    import fitz  # PyMuPDF
    import numpy as np
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError as e:
    st.title("Text Document Summarizer")
    st.error(f"A required library is missing: '{e.name}'")
    st.warning(
        "To continue, please install the required packages by running the "
        "following command in your terminal:"
    )
    st.code("pip install -r requirements.txt", language="bash")
    st.stop()  # Pause the app until dependencies are installed

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
    nltk.download("punkt")

def read_text_file(uploaded_file):
    """Reads the content of a text file."""
    try:
        # Read the raw bytes first
        raw_bytes = uploaded_file.read()
        # Try decoding with utf-8, then fall back to latin-1 if it fails.
        try:
            return raw_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return raw_bytes.decode('latin-1')
    except Exception as e:
        st.error(f"Error reading TXT file: {e}")
        return ""

def read_docx_file(uploaded_file):
    """Reads the content of a .docx file."""
    try:
        document = Document(uploaded_file)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return ""

def read_pdf_file(uploaded_file):
    """Reads the content of a .pdf file using PyMuPDF."""
    try:
        # PyMuPDF works with bytes, so we read the uploaded file into memory
        file_bytes = uploaded_file.read()
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

def create_pdf_from_text(text):
    """Creates a PDF in memory from a string of text."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # Encode text to latin-1, replacing unsupported characters to prevent errors.
    # FPDF's core fonts have limited Unicode support. For full Unicode, a custom
    # font would need to be added with `pdf.add_font()`.
    encoded_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, encoded_text)
    # Return the PDF content as bytes.
    return pdf.output()


def _get_download_data(summary_text, output_format):
    """Prepares the file data, extension, and MIME type for downloading."""
    if output_format == "PDF":
        file_data = create_pdf_from_text(summary_text)
        file_extension = "pdf"
        mime_type = "application/pdf"
    else:  # Text (.txt)
        file_data = summary_text.encode('utf-8')
        file_extension = "txt"
        mime_type = "text/plain"
    return file_data, file_extension, mime_type

def summarize_text(text, num_sentences=3):
    """
    Generates a concise summary using the TextRank algorithm.

    This function models sentences as a graph and uses their similarity
    to rank them, extracting the most important ones to form a summary.
    """
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text  # Return original text if it's too short to summarize

    stop_words = set(stopwords.words("english"))

    # Create TF-IDF vectors for sentences
    vectorizer = TfidfVectorizer(stop_words=list(stop_words))
    try:
        # Use a preprocessor to remove non-alphanumeric characters for better vectorization
        preprocessed_sentences = [s.lower() for s in sentences]
        sentence_vectors = vectorizer.fit_transform(preprocessed_sentences)
    except ValueError:
        # Occurs if the vocabulary is empty (e.g., only stop words)
        return "Could not generate a summary. The document may not contain enough meaningful content."

    # Calculate the similarity matrix (adjacency matrix of the graph)
    # The matrix should be square
    similarity_matrix = cosine_similarity(sentence_vectors, sentence_vectors)

    # Use the PageRank algorithm to rank sentences
    scores = np.zeros(len(sentences))
    damping_factor = 0.85
    epsilon = 1.0e-5
    iterations = 100
    converged = False

    for _ in range(iterations):
        prev_scores = np.copy(scores)
        for i in range(len(sentences)):
            summation = 0
            for j in range(len(sentences)):
                if i == j:
                    continue
                # Normalize by the sum of outgoing connections for node j
                denominator = sum(similarity_matrix[j])
                if denominator > 0:
                    summation += (similarity_matrix[i][j] * prev_scores[j]) / denominator

            scores[i] = (1 - damping_factor) + damping_factor * summation
        if np.allclose(prev_scores, scores, atol=epsilon):
            converged = True
            break

    # Get the top N sentences based on scores
    summary_sentences = heapq.nlargest(
        num_sentences, range(len(scores)), key=scores.__getitem__
    )

    # Reorder summary sentences based on their original appearance
    summary_sentences.sort()
    summary = " ".join([sentences[i] for i in summary_sentences])
    return summary if summary else "Could not generate a summary."

def generate_report(original_texts, summaries, filenames):
    """Generates a report and saves it to the reports directory."""
    report_filename = os.path.join("reports", f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_filename, "w") as f:
        f.write("--- Multi-Document Summary Report ---\n\n")
        f.write(f"Number of docs checked: {st.session_state.usage_count}\n")
        f.write(f"Billing summary: {st.session_state.usage_count * 10} INR\n")
        f.write("\n" + "="*40 + "\n")

        for i, (text, summary, filename) in enumerate(zip(original_texts, summaries, filenames)):
            f.write(f"\n--- Document {i+1}: {filename} ---\n")
            f.write("\n--- Original Text ---\n")
            f.write(text)
            f.write("\n\n--- Generated Summary ---\n")
            f.write(summary)
            f.write("\n" + "="*40 + "\n")
    return report_filename


def display_billing_summary():
    st.write(f"Docs Checked: {st.session_state.usage_count}")
    st.write(f"Total Bill: {st.session_state.usage_count * 10} INR")

def simulate_text_summary():
    """Simulates summarizing a text document."""
    sample_text = (
        "The James Webb Space Telescope is a space telescope designed primarily to "
        "conduct infrared astronomy. As the largest optical telescope in space, its "
        "greatly improved infrared resolution and sensitivity allow it to view objects "
        "too old, distant, or faint for the Hubble Space Telescope."
    )
    st.subheader("Simulated Text Summary:")
    st.write("**Original Text:**", sample_text)

    summary = summarize_text(sample_text)
    st.write("Summary:")
    st.write(summary)

def process_uploaded_files(uploaded_files):
    """
    Extracts documents from various file types (DOCX, ZIP) and returns a list
    of file-like objects ready for processing.
    """
    docs_to_process = []
    MAX_FILE_SIZE_IN_ZIP = 10 * 1024 * 1024  # 10 MB limit for files inside a ZIP
    SUPPORTED_EXTENSIONS = ('.docx', '.txt', '.pdf')

    for uploaded_file in uploaded_files:
        file_type = uploaded_file.type
        if file_type in [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # .docx
            "text/plain", # .txt
            "application/pdf" # .pdf
        ]:
            # It's a supported document type, add it directly
            docs_to_process.append(uploaded_file)
        elif file_type == "application/zip":
            # It's a ZIP file, extract supported documents from it securely
            try:
                with zipfile.ZipFile(uploaded_file) as z:
                    for member in z.infolist():
                        # Process supported file types, ignore directories
                        if member.filename.endswith(SUPPORTED_EXTENSIONS) and not member.is_dir():
                            # Security Check: Prevent Zip Bomb attacks
                            if member.file_size > MAX_FILE_SIZE_IN_ZIP:
                                st.warning(f"Skipping '{member.filename}' in '{uploaded_file.name}' because it is too large ({member.file_size / 1024**2:.1f} MB).")
                                continue
                            
                            with z.open(member) as inner_file:
                                # Wrap the extracted file in a file-like object
                                file_in_memory = io.BytesIO(inner_file.read())
                                file_in_memory.name = os.path.basename(member.filename)
                                docs_to_process.append(file_in_memory)
            except zipfile.BadZipFile:
                st.error(f"The uploaded ZIP file '{uploaded_file.name}' is corrupted or invalid.")
            except Exception as e:
                st.error(f"An error occurred while processing '{uploaded_file.name}': {e}")
        else:
            st.warning(f"Unsupported file type ('{uploaded_file.name}'). This file will be ignored.")

    return docs_to_process

def main():
    # Initialize session state for usage counter
    if 'usage_count' not in st.session_state:
        st.session_state.usage_count = 0

    st.title("Multi-Document Summarizer")
    st.write(
        "Upload 1 to 3 documents for summarization. "
        "Supported formats are `.docx`, `.txt`, `.pdf`, and `.zip` archives containing these files."
    )

    # Ensure the 'reports' directory exists
    if not os.path.exists("reports"):
        os.makedirs("reports")

    # File upload section
    uploaded_files = st.file_uploader(
        "Upload documents or ZIP archives",
        type=["docx", "txt", "pdf", "zip"],
        accept_multiple_files=True
    )

    if uploaded_files:
        docs_to_process = process_uploaded_files(uploaded_files)

        num_docs = len(docs_to_process)

        # Validate that the number of documents is within the allowed range (1 to 3).
        if not 1 <= num_docs <= 3:
            st.error(f"Found {num_docs} documents. Please upload between 1 and 3 supported documents.")
            st.stop()

        # Increment usage counter for the batch of 3 documents
        st.session_state.usage_count += len(docs_to_process)

        # Add a slider to control summary length for all documents
        num_sentences = st.slider("Summary Length (number of sentences):", min_value=1, max_value=10, value=3)

        # Add radio buttons for selecting the output format
        output_format = st.radio("Select Download Format:", ("Text (.txt)", "PDF"), horizontal=True)

        st.divider()

        # Create three columns for side-by-side display
        cols = st.columns(num_docs)
        
        all_texts = []
        all_summaries = []

        for i, doc_file in enumerate(docs_to_process):
            with cols[i]:
                with st.container(border=True):
                    st.subheader(f"Document {i+1}")
                    st.caption(doc_file.name)
                    
                    # Choose the correct reader based on the file extension
                    if doc_file.name.endswith('.docx'):
                        raw_text = read_docx_file(doc_file)
                    elif doc_file.name.endswith('.txt'):
                        raw_text = read_text_file(doc_file)
                    elif doc_file.name.endswith('.pdf'):
                        raw_text = read_pdf_file(doc_file)

                    all_texts.append(raw_text)

                    if raw_text:
                        summary_text = summarize_text(raw_text, num_sentences=num_sentences)
                        all_summaries.append(summary_text)
                        st.info(f"**Summary:**\n\n{summary_text}")

                        file_data, file_extension, mime_type = _get_download_data(
                            summary_text, output_format
                        )

                        st.download_button(
                            label=f"Download as .{file_extension}",
                            data=file_data,
                            file_name=f"summary_{os.path.splitext(doc_file.name)[0]}.{file_extension}",
                            mime=mime_type,
                            key=f"download_button_{i}" # Unique key is crucial for multiple buttons
                        )


        st.divider()
        st.subheader("Billing Summary:")
        display_billing_summary()

        if st.button("Generate Report"):
            report_file = generate_report(all_texts, all_summaries, [f.name for f in docs_to_process])
            st.success(f"Report saved to {report_file}")

    if st.button("Simulate Text Summary"):
        simulate_text_summary()

if __name__ == "__main__":
    main()
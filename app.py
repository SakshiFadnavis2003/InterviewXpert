from flask import Flask, request, render_template
import cv2
import mediapipe as mp
import numpy as np
import os
from flask import Flask, render_template, request, jsonify
import os
import speech_recognition as sr
import google.generativeai as genai
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Configure the Generative AI model
logging.basicConfig(level=logging.DEBUG)
os.environ["API_KEY"] = 'AIzaSyDY1U-9SQtnOrmCTBwpi5nrfARznuBUEzw'
genai.configure(api_key=os.environ["API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")
UPLOAD_FOLDER = r"C:\Users\tusha\OneDrive\Desktop\InterviewXpert\uploads"  # Folder to store uploaded files
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def generate_questions(prompt):
    response = model.generate_content(prompt)
    questions = response.text.split("\n")

    filtered_questions = []
    for line in questions:
        if line.strip() and line[0].isdigit():  # Check if line starts with a number
            filtered_questions.append(line.split(". ", 1)[-1].strip())

    final_questions = ["Introduce yourself"] + filtered_questions
    numbered_questions = "\n".join([f"{i + 1}. {q}" for i, q in enumerate(final_questions)])
    return numbered_questions

@app.route('/index2', methods=['GET', 'POST'])
def index2():
    generated_output = ""
    if request.method == 'POST':
        prompt = """Generate a list of 10 interview questions specifically tailored for a Data Analyst role. The questions should cover various aspects of the role, including:

        Technical Proficiency:
        - Questions related to data analysis using Python, SQL, and Excel.
        - Queries on data visualization tools like Tableau, Power BI, and Matplotlib.
        - Concepts of data cleaning, manipulation, and wrangling (Pandas, NumPy).
        - Statistical analysis techniques, including regression, classification, and hypothesis testing.
        - Basic machine learning concepts relevant to data analysis (e.g., clustering, decision trees).

        Behavioral Skills:
        - Questions to assess problem-solving and analytical thinking.
        - Communication skills, particularly in explaining technical concepts to non-technical stakeholders.
        - Experience in collaboration, teamwork, and dealing with feedback.
        - Time management and multi-tasking capabilities.

        Scenario-Based Queries:
        - Hypothetical situations that test real-time data-driven decision-making.
        - Case studies involving data cleaning, feature engineering, and trend analysis.
        - Questions requiring explanation of how to derive insights from raw data.
        - Business-oriented scenarios that assess understanding of metrics, KPIs, and ROI.

        Ensure that the questions vary in difficulty and include a mix of technical and soft skill assessments. Each question should be concise and specific, providing a clear idea of what aspect of the role it aims to evaluate."""

        generated_output = generate_questions(prompt)

    return render_template('index2.html', output=generated_output)

def generate_feedback(transcription):
    prompt = f"Based on the following transcription of an interview response, provide constructive feedback:\n\nTranscription:\n{transcription}\n\nQuestion:\n{question}"
    response = model.generate_content(prompt)
    return response.text


@app.route('/question/<int:question_id>')
def question_page(question_id):
    prompt = """Generate a list of 10 interview questions specifically tailored for a Data Analyst role. The questions should cover various aspects of the role, including:

        Technical Proficiency:
        - Questions related to data analysis using Python, SQL, and Excel.
        - Queries on data visualization tools like Tableau, Power BI, and Matplotlib.
        - Concepts of data cleaning, manipulation, and wrangling (Pandas, NumPy).
        - Statistical analysis techniques, including regression, classification, and hypothesis testing.
        - Basic machine learning concepts relevant to data analysis (e.g., clustering, decision trees).

        Behavioral Skills:
        - Questions to assess problem-solving and analytical thinking.
        - Communication skills, particularly in explaining technical concepts to non-technical stakeholders.
        - Experience in collaboration, teamwork, and dealing with feedback.
        - Time management and multi-tasking capabilities.

        Scenario-Based Queries:
        - Hypothetical situations that test real-time data-driven decision-making.
        - Case studies involving data cleaning, feature engineering, and trend analysis.
        - Questions requiring explanation of how to derive insights from raw data.
        - Business-oriented scenarios that assess understanding of metrics, KPIs, and ROI.

        Ensure that the questions vary in difficulty and include a mix of technical and soft skill assessments. Each question should be concise and specific, providing a clear idea of what aspect of the role it aims to evaluate."""

    questions = generate_questions(prompt).split("\n")

    if 0 <= question_id < len(questions):
        question = questions[question_id]
    else:
        question = "Invalid Question"

    return render_template('question.html', question=question)


def convert_to_wav(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    wav_path = audio_file_path.replace(os.path.splitext(audio_file_path)[1], ".wav")
    audio.export(wav_path, format="wav")
    return wav_path

def convert_audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)

@app.route('/audio_to_text', methods=['POST'])
def audio_to_text():
    if 'audio' not in request.files:
        return "No audio file provided", 400

    audio_file = request.files['audio']
    
    # Check the file type
    print("Uploaded file type:", audio_file.content_type)

    # Save the uploaded audio file
    audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(audio_path)

    try:
        # Convert audio to WAV if necessary
        if not audio_path.endswith('.wav'):
            wav_path = convert_to_wav(audio_path)  # Convert to WAV
        else:
            wav_path = audio_path

        # Transcribe the audio
        transcription = convert_audio_to_text(wav_path)
        return jsonify({"transcription": transcription})

    except Exception as e:
        print("Error during transcription:", str(e))
        return jsonify({"error": str(e)}), 500


def generate_feedback(transcription):
    # Feedback generation prompt
    prompt = f"Please modify the transcription in understandable and readable manner and remove the repeated words and then analyze the transcription of the interview for the candidate applying for the Data Analyst position and provide comprehensive feedback.Assess their technical knowledge by examining their understanding of relevant concepts, their ability to explain technical terms, and their responses to technical questions as presented in the transcription. Furthermore, analyze their problem-solving skills by reviewing their approach to situational or technical questions, as well as their critical thinking and reasoning abilities demonstrated in their answers. Finally, offer an overall impression of the candidate, highlighting strengths and areas for improvement, and comment on their suitability for the role based on the interview performance.:\n\nTranscription:\n{transcription} "
    
    # Generate feedback using the generative model
    response = model.generate_content(prompt)
    return response.text

@app.route('/templates/feedback.html')
def feedback():
    return render_template('templates/feedback.html')

# Remove the redundant `generate_feedback()` function

@app.route('/submit', methods=['POST'])
def submit():
    transcription = request.form.get('transcription')
    feedback = generate_feedback(transcription)
    return jsonify({'transcription': transcription, 'feedback': feedback})


@app.route('/analyze_body_language', methods=['POST'])
def analyze_body_language_endpoint():
    video_file = request.files.get('video')
    if video_file:
        if video_file.content_type not in ['video/webm', 'video/mp4']:  # Validate file type
            return jsonify({"error": "Invalid video format. Please upload a WebM or MP4 file."}), 400

        video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))
        video_file.save(video_path)

        body_language_feedback = analyze_body_language(video_path)
        return jsonify(body_language_feedback)
    else:
        return jsonify({"error": "No video file provided."}), 400


# Initialize MediaPipe models
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def analyze_body_language(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video file. Check if the file is valid."}
        
        feedback = {"face_touches": 0, "hand_gestures": []}
        overall_score = 100
        frame_counter = 0
        frame_skip = 5  # Process every 5th frame
        face_touch_last_frame = -10

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit the loop if no more frames are available

            frame_counter += 1
            if frame_counter % frame_skip != 0:
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            hand_results = hands.process(image)
            face_mesh_results = face_mesh.process(image)

            # Face Touch Analysis
            if face_mesh_results.multi_face_landmarks and hand_results.multi_hand_landmarks:
                face_touch_detected = False
                for face_landmarks in face_mesh_results.multi_face_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        for hand_landmark in hand_landmarks.landmark:
                            for face_landmark in face_landmarks.landmark:
                                distance = np.sqrt(
                                    (hand_landmark.x - face_landmark.x) ** 2 +
                                    (hand_landmark.y - face_landmark.y) ** 2
                                )
                                if distance < 0.05:
                                    face_touch_detected = True
                                    break

                    if face_touch_detected:
                        break

                if face_touch_detected and frame_counter - face_touch_last_frame > 10:
                    feedback["face_touches"] += 1
                    face_touch_last_frame = frame_counter
                    if len(feedback["hand_gestures"]) == 0:
                        feedback["hand_gestures"].append(
                            "Mistake: Face touch detected. "
                            "Correction: Try to avoid touching your face."
                        )
                    overall_score -= 5

        cap.release()
        cv2.destroyAllWindows()

        feedback["overall_score"] = max(0, int(overall_score))
        feedback["summary"] = (
            f"Overall Body Language Score: {feedback['overall_score']}%\n"
            f"Face touched: {feedback['face_touches']} times.\n"
            "Remember to avoid scratching or rubbing your face for a professional appearance.\n"
            f"Face Touch Count: {feedback['face_touches']}\n"
            f"Hand Gestures Feedback: {feedback['hand_gestures']}\n"
            f"Overall Score: {feedback['overall_score']}\n"
        )
        
        return feedback

    except Exception as e:
        print(f"Error analyzing video: {e}")
        return {"error": str(e)}



if __name__ == '__main__':
    app.run(debug=True)
# InterviewXpert

**InterviewXpert** is an AI-based virtual interview preparation platform designed to help users practice for job interviews with a focus on both verbal and non-verbal communication. The platform uses generative AI for dynamic question generation, speech-to-text for interactive conversation, and computer vision for body language analysis. This tool provides a comprehensive feedback mechanism to improve overall interview performance.

---

## Methodology

The development of **InterviewXpert** was structured around integrating cutting-edge AI technologies, including Natural Language Processing (NLP), generative AI, and computer vision, to create an immersive user experience. The platform was developed step-by-step to ensure a robust, interactive system that helps users enhance their interview skills.


![image](https://github.com/user-attachments/assets/cc91576c-2506-4a5f-89c4-5a22c8584a65)


### 1. Initial Planning and Technology Integration

The project began with a clear blueprint to integrate AI technologies seamlessly into the platform. The goal was to leverage advanced pre-existing AI services for rapid development while ensuring a high level of customization. Python was selected as the primary programming language due to its extensive libraries and ease of integration with third-party AI tools. Key libraries used include Google’s generative AI models for NLP tasks, OpenCV, and MediaPipe for body language analysis.

The first stage involved setting up the environment and installing necessary libraries to ensure compatibility with AI tools like Google's generative models and OpenCV’s video processing capabilities.

### 2. Model Development

#### i. Generative AI for Question Creation
To simulate real-world interviews, the **Gemini 1.5 Flash** model from Google was integrated to generate role-specific interview questions. These questions ranged from general competency inquiries to detailed, job-specific questions, providing a highly personalized interview experience. 

#### ii. Real-Time Interaction with Speech Processing
For an interactive experience, **Speech-to-Text (STT)** technology was integrated to transcribe user responses. Additionally, **Text-to-Speech (TTS)** technology was used to read out questions, creating a conversational feel. The system enabled real-time communication between the user and the platform, allowing a seamless interview practice environment.

#### iii. Advanced Body Language Analysis
To incorporate non-verbal cues, **OpenCV** and **MediaPipe** were used to analyze body language. MediaPipe’s holistic model was leveraged to track facial landmarks, body pose, and hand movements in real time, providing feedback on posture, nervousness, and other behavioral signals.

#### iv. Constructive Feedback Mechanism
The platform’s feedback system analyzed both verbal and non-verbal cues. By processing user responses and evaluating body language, the system provided actionable insights regarding the content quality, coherence, and engagement level of the responses. This feedback helped users identify areas for improvement.

### 3. Website Creation and Model Integration

A responsive website was created using **HTML**, **CSS**, and **JavaScript** to host the AI models and provide a user-friendly interface. The site was structured to facilitate easy navigation through various interview stages.


![image](https://github.com/user-attachments/assets/8f7422bb-7d50-4471-b392-842f3d468fbc)


#### i. Question Generation Interface
A button initiates the question generation model, displaying a list of relevant interview questions. Each question is clickable, allowing users to proceed to the next stage.


![image](https://github.com/user-attachments/assets/aacbded7-3944-48c7-8583-4121fe9c2d3c)


#### ii. Interview Interaction
Once a question is selected, the user is directed to a new page where the question is displayed. The platform uses **TTS** to read out the question, simulating a more interactive interview experience.


![image](https://github.com/user-attachments/assets/018463df-f0b5-4577-ac59-1f7e06369156)


#### iii. Response Recording and Processing
Users are prompted to record their answers, which are captured and converted into text using **speech-to-text** technology. The system then processes the response to generate feedback.

#### iv. Feedback Display
After each response, detailed feedback is provided, including insights on content quality, coherence, and engagement. Body language analysis feedback is also displayed, focusing on posture, hand movements, and signs of stress or confidence.


![image](https://github.com/user-attachments/assets/bbef8d65-eecf-4930-974e-4edc7f1862be)


#### v. Iterative Practice
Users can repeat the process with additional questions, allowing them to refine both verbal and non-verbal responses. The iterative practice helps users prepare comprehensively for real-life interviews.

---

## Features

- **Dynamic Question Generation**: Role-specific, job-related questions generated by Google’s **Gemini 1.5 Flash** model.
- **Speech-to-Text Integration**: Real-time transcription of responses for further analysis.
- **Text-to-Speech Functionality**: The platform reads out questions, enhancing the interactive experience.
- **Body Language Analysis**: Integration of **OpenCV** and **MediaPipe** to assess non-verbal cues such as posture, facial expressions, and hand movements.
- **Detailed Feedback**: Constructive feedback based on both verbal content and non-verbal behavior.
- **Iterative Practice**: Ability to repeat the interview with new questions and improve over time.

---

## Technologies Used

- **Python**: Primary language for backend development and AI model integration.
- **Google Gemini 1.5 Flash**: Generative AI model for question creation.
- **Speech-to-Text (STT)**: For transcribing user responses.
- **Text-to-Speech (TTS)**: To read out the generated questions.
- **OpenCV**: For computer vision and body language analysis.
- **MediaPipe**: For detecting facial landmarks and tracking body pose and hand movements.
- **HTML, CSS, JavaScript**: For building the interactive website interface.

---

## Setup and Installation

### Requirements

- Python 3.7 or higher
- Libraries:
  - `opencv-python`
  - `mediapipe`
  - `google-cloud` (for integrating Google AI)
  - `speech_recognition`
  - `gTTS` (for text-to-speech functionality)

### Installation Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/InterviewXpert.git
    cd InterviewXpert
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    python app.py
    ```

4. Access the website in your browser at `localhost:5000`.

---

## Contributors

- [@SakshiFadnavis2003](https://github.com/SakshiFadnavis2003)
- [@KhushiBajpai2003](https://github.com/KhushiBajpai2003)
- [@shrutiijainn11](https://github.com/shrutiijainn11)
  
---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

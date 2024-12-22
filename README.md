# Image Captioning Web Application

This repository hosts an Image Captioning Web Application that uses a pre-trained model to generate descriptive captions for uploaded images. The project includes a backend built with Flask and a frontend using HTML.

### Note: I have created the folder in two parts, if you want to run this code you need to first host the model on google colab for API then you need to use image_caption.zip for webpage.
## Features

- **Image Upload**: Users can upload image files via the web interface.
- **Caption Generation**: Automatically generates a descriptive caption for the uploaded image using a pre-trained model.
- **API Integration**: Communicates with a backend model hosted on Google Colab via an ngrok URL.
- **Dynamic Results**: Displays the generated caption in real time on the web page.
- **File Validation**: Ensures only valid image formats (e.g., PNG, JPEG) are processed.

## Components

### 1. Backend (Flask Application)

**File**: `app.py`

#### Key Features:
- Handles file uploads and validations
- Communicates with an external API for caption generation
- Processes responses and returns captions to the frontend

#### Endpoints:
- `/`: Renders the homepage
- `/upload`: Handles image uploads and caption generation

### 2. Frontend

**File**: `index.html`

#### Key Features:
- Simple and user-friendly interface for uploading images
- JavaScript handles file submission and displays results dynamically
- Minimal styling for clean presentation

### 3. Hosted Model

- **Location**: Hosted on Google Colab
- **Framework**: HuggingFace VisionEncoderDecoderModel for image captioning
- **Model Name**: nlpconnect/vit-gpt2-image-captioning
- **Tunnel**: Exposed via ngrok for external API access

## Installation and Setup

### Prerequisites:
- Python 3.8+
- Flask
- ngrok for exposing local services (optional if using another hosting method)

### Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-captioning-webapp.git
cd image-captioning-webapp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the Flask server:
```bash
python app.py
```

4. Launch the frontend:
   - Open `http://127.0.0.1:5000` in your browser

5. (Optional) Host the backend model:
   - Run the backend Python code on Google Colab
   - Set up ngrok to expose the backend endpoint

6. Update the API URL:
   - Replace the placeholder API URL in `app.py` and `index.html` with your ngrok URL

## Usage

1. Open the web application in your browser
2. Upload an image file
3. Wait for the application to generate a caption
4. View the caption displayed on the result page

## Project Workflow

1. **Image Upload**: User selects an image file
2. **Backend Processing**:
   - Flask validates the file
   - The image is sent to the hosted model API
   - The API generates a caption
3. **Frontend Display**:
   - The generated caption is dynamically shown on the web page

## File Structure

```
.
├── app.py               # Backend Flask application
├── templates
│   └── index.html       # Frontend HTML file
├── static
│   └── uploads          # Directory for uploaded images
└── requirements.txt     # Dependencies for the project
```

## Future Enhancements

- **Enhanced Styling**: Use CSS frameworks like Bootstrap or Tailwind CSS
- **Multiple Uploads**: Support batch processing of images
- **Improved Hosting**: Transition from ngrok to a dedicated cloud service (e.g., AWS, GCP)
- **Additional Features**:
  - Caption history
  - Download options for generated captions


## Acknowledgments

- **Model**: nlpconnect/vit-gpt2-image-captioning from HuggingFace
- **Libraries**: Flask, PIL, Requests
- **Tools**: ngrok, Google Colab

## Contact

For questions or feedback, contact:
- **Name**: Shubham Prakash
- **Email**: shubhamprakash2025@gmail.com

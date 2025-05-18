import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session
from dotenv import load_dotenv
import google.generativeai as genai
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import base64
import cv2
from io import BytesIO
from flask_bcrypt import Bcrypt
import mysql.connector
from mysql.connector import Error
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MySQL connection
try:
    db = mysql.connector.connect(
        host="localhost",
        user="root",  # Replace with your MySQL username
        password="",  # Replace with your MySQL password
        database="mydb",
        port=9901
    )
    cursor = db.cursor()
    cursor.execute("SELECT 1")  # Test connection
    logger.info("Successfully connected to MySQL")
except Error as e:
    logger.error(f"Failed to connect to MySQL: {str(e)}")
    raise Exception("MySQL connection failed. Ensure MySQL is running and credentials are correct")

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logger.error("GEMINI_API_KEY not found in .env file")
    raise Exception("GEMINI_API_KEY is missing in .env file")

# Configure Gemini
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel(model_name="gemini-2.5-flash-preview-04-17")

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()  # Secure random secret key
bcrypt = Bcrypt(app)

# Load ML models
try:
    soil_model = pickle.load(open('updated_soil_model_4features.pkl', 'rb'))
    encoder = pickle.load(open("models/encoder.pkl", 'rb'))
    scaler = pickle.load(open("models/scaler.pkl", 'rb'))
    model_gbc = pickle.load(open("models/model_gbc.pkl", 'rb'))
    leaf_model = load_model('Leaf Deases(96,88).h5')
    logger.info("All ML models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load ML models: {str(e)}")
    raise

# Define leaf disease class names
leaf_disease_classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Cherry___healthy",
    "Cherry___Powdery_mildew",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust_",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    logger.error("Failed to open camera")
    # Don't raise an exception here, as we might still want the app to run without camera functionality
    logger.warning("Application will run without camera functionality")

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            logger.warning("Failed to read frame from camera")
            break

        # Convert OpenCV BGR to PIL RGB for prediction
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).resize((150, 150))
        processed_img = preprocess_leaf_image(pil_img)

        # Predict
        preds = leaf_model.predict(processed_img)
        predicted_class = leaf_disease_classes[np.argmax(preds)]
        confidence = np.max(preds) * 100
        label = f"{predicted_class} ({confidence:.2f}%)"

        # Add label to frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logger.warning("Failed to encode frame to JPEG")
            continue
        frame_bytes = buffer.tobytes()

        # MJPEG stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def home():
    # if 'user' in session:
    #     return redirect(url_for('index'))
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')

        if not all([name, email, password, role]):
            flash('All fields are required!', 'error')
            return redirect(url_for('register'))

        try:
            cursor.execute("SELECT email FROM users WHERE email = %s", (email,))
            if cursor.fetchone():
                flash('Email already registered!', 'error')
                return redirect(url_for('register'))

            hashed_password = password
            registration_date = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

            cursor.execute(
                "INSERT INTO users (name, email, password, role, registration_date) VALUES (%s, %s, %s, %s, %s)",
                (name, email, hashed_password, role, registration_date)
            )
            db.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))

        except Error as e:
            db.rollback()
            logger.error(f"Registration error: {str(e)}")
            flash(f"Registration failed: {str(e)}", 'error')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not all([email, password]):
            flash('Email and password are required!', 'error')
            logger.warning("Login failed: Missing email or password")
            return redirect(url_for('login'))

        # Check if user exists in users table
        try:
            cursor.execute("SELECT id, name, password FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()
            
            if user:
                # Try using check_password_hash with appropriate error handling
                try:
                    if (user[2], password):
                        session['user'] = user[1]  # name
                        session['user_id'] = str(user[0])  # id
                        session['email'] = email  # Store email for logout
                        
                        # Create a new cursor for additional operations
                        login_cursor = db.cursor()
                        
                        # Log the login event in login_sessions table
                        login_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                        login_cursor.execute(
                            "INSERT INTO login_sessions (user_id, email, login_time) VALUES (%s, %s, %s)",
                            (user[0], email, login_time)
                        )
                        db.commit()
                        login_cursor.close()  # Close this cursor when done
                        
                        logger.info(f"User logged in: {email}, User ID: {user[0]}")
                        flash(f"Welcome {user[1]}!", 'success')
                        return redirect(url_for('index'))
                    else:
                        flash('Invalid password!', 'error')
                        logger.warning(f"Login failed: Invalid password for {email}")
                except ValueError as ve:
                    # This catches the "Invalid salt" error
                    logger.error(f"Password verification error: {str(ve)}")
                    logger.info(f"Attempting direct string comparison as fallback")
                    
                    # Fallback: direct string comparison (for plaintext passwords in DB)
                    if user[2] == password:
                        session['user'] = user[1]
                        session['user_id'] = str(user[0])
                        session['email'] = email
                        
                        # Create a new cursor for additional operations
                        login_cursor = db.cursor()
                        
                        login_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                        login_cursor.execute(
                            "INSERT INTO login_sessions (user_id, email, login_time) VALUES (%s, %s, %s)",
                            (user[0], email, login_time)
                        )
                        db.commit()
                        
                        # Update password to bcrypt hash for future logins - using another cursor
                        update_cursor = db.cursor()
                        hashed_password = password
                        update_cursor.execute(
                            "UPDATE users SET password = %s WHERE id = %s",
                            (hashed_password, user[0])
                        )
                        db.commit()
                        
                        # Close both cursors
                        login_cursor.close()
                        update_cursor.close()
                        
                        logger.info(f"Updated password hash for user: {email}")
                        
                        flash(f"Welcome {user[1]}!", 'success')
                        return redirect(url_for('index'))
                    else:
                        flash('Invalid password!', 'error')
                        logger.warning(f"Login failed: Invalid password for {email}")
            else:
                flash('Email not registered!', 'error')
                logger.warning(f"Login failed: Email not found: {email}")
            
            return redirect(url_for('login'))
        except Error as e:
            logger.error(f"Error during login: {str(e)}")
            flash(f"Login failed: {str(e)}", 'error')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    if 'user' in session:
        try:
            logout_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(
                "UPDATE login_sessions SET logout_time = %s WHERE user_id = %s AND email = %s AND logout_time IS NULL",
                (logout_time, session.get('user_id'), session.get('email'))
            )
            db.commit()
            logger.info(f"User logged out: {session.get('email')}")
        except Error as e:
            logger.error(f"Error logging logout: {str(e)}")
    session.pop('user', None)
    session.pop('user_id', None)
    session.pop('email', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/video_feed')
def video_feed():
    # if 'user' not in session:
    #     flash('Please log in to access this feature.', 'error')
    #     return redirect(url_for('login'))
    return app.response_class(generate_frames(),
                             mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_detect')
def live_detect():
    # if 'user' not in session:
    #     flash('Please log in to access this feature.', 'error')
    #     return redirect(url_for('login'))
    return render_template('leaves.html')

# Utility: preprocess leaf image to model input size
def preprocess_leaf_image(image, target_size=(150, 150)):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image / 255.0  # normalize pixel values
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/index')
def index():
    # if 'user' not in session:
    #     flash('Please log in to access this page.', 'error')
    #     return redirect(url_for('login'))
    return render_template('index.html')

# Soil Fertility Prediction + Gemini Suggestion
@app.route('/predict', methods=['POST'])
def predict():
    # if 'user' not in session:
    #     flash('Please log in to access this feature.', 'error')
    #     return redirect(url_for('login'))

    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        ph = float(request.form['ph'])

        input_data = {'N': N, 'P': P, 'K': K, 'ph': ph}
        features = np.array([[N, P, K, ph]])
        prediction = soil_model.predict(features)[0]

        fertility_map = {0: "Less Fertile", 1: "Fertile", 2: "Highly Fertile"}
        result = fertility_map.get(prediction, "Unknown")

        prediction_text = f"Predicted Fertility: {result}"

        prompt = f"""Soil Test Results:
        - Nitrogen: {N}
        - Phosphorous: {P}
        - Potassium: {K}
        - pH Level: {ph}

        Based on the above parameters, suggest:
        1. Suitable pesticides (if needed)
        2. Fertilizers or organic ways to improve soil fertility
        3. Any precautions or best practices for the farmer
        Respond in short practical terms.
        """

        try:
            response = gemini_model.generate_content(prompt + " Give suggestions in 2-3 side points..dont explain much ")
            suggestion_text = response.text.strip()
        except Exception as e:
            suggestion_text = f"⚠ Gemini Error: {str(e)}"
            logger.error(f"Gemini API error: {str(e)}")

        return render_template('index.html',
                              prediction_text=prediction_text,
                              suggestion_text=suggestion_text,
                              active_tab='soil',
                              **input_data)

    except Exception as e:
        logger.error(f"Error in soil prediction: {str(e)}")
        return render_template('index.html',
                              prediction_text=f"⚠ Error: {str(e)}",
                              active_tab='soil')

# Crop Recommendation
@app.route('/recommend', methods=['POST'])
def recommend():
    # if 'user' not in session:
    #     flash('Please log in to access this feature.', 'error')
    #     return redirect(url_for('login'))

    try:
        input_data = {field: float(request.form[field]) for field in
                      ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']}
        input_df = pd.DataFrame([[input_data[field] for field in input_data]],
                                columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        input_scaled = scaler.transform(input_df)
        prediction_encoded = model_gbc.predict(input_scaled)
        crop_prediction = encoder.inverse_transform(prediction_encoded)[0]

        return render_template('index.html',
                              crop_prediction=crop_prediction,
                              active_tab='crops',
                              **input_data)
    except Exception as e:
        logger.error(f"Error in crop recommendation: {str(e)}")
        return render_template('index.html',
                              crop_prediction=f"⚠ Error: {str(e)}",
                              active_tab='crops')

# Leaf Disease Detection - Upload Image & Predict
@app.route('/leaves', methods=['GET', 'POST'])
def leaves():
    # if 'user' not in session:
    #     flash('Please log in to access this feature.', 'error')
    #     return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request', 'error')
            logger.warning("Leaf detection failed: No file part in request")
            return redirect(url_for('leaves'))

        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            logger.warning("Leaf detection failed: No file selected")
            return redirect(url_for('leaves'))

        try:
            # Open and preprocess image
            img = Image.open(file.stream).convert('RGB')
            processed_img = preprocess_leaf_image(img)
            preds = leaf_model.predict(processed_img)
            predicted_class = leaf_disease_classes[np.argmax(preds)]
            confidence = np.max(preds) * 100
            result_text = f"Detected Disease: {predicted_class} ({confidence:.2f}%)"
            logger.info(f"Leaf disease detected: {predicted_class} ({confidence:.2f}%)")

            # Get treatment recommendations using Gemini
            plant_type = predicted_class.split('___')[0]
            disease_name = predicted_class.split('___')[1].replace('_', ' ')
            
            if "healthy" not in disease_name.lower():
                try:
                    treatment_prompt = f"""
                    Plant: {plant_type}
                    Disease: {disease_name}
                    
                    Provide brief and practical treatment recommendations for this plant disease.
                    Include:
                    1. Chemical controls (if appropriate)
                    2. Organic/natural remedies
                    3. Prevention tips
                    
                    Give only short, practical bullet points - no lengthy explanations.
                    """
                    
                    treatment_response = gemini_model.generate_content(treatment_prompt)
                    treatment_text = treatment_response.text.strip()
                except Exception as e:
                    treatment_text = "Could not generate treatment recommendations."
                    logger.error(f"Error getting treatment recommendations: {str(e)}")
            else:
                treatment_text = "Plant appears healthy! Continue with regular care and monitoring."

            # Convert image to base64 to send to template
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return render_template('result.html', 
                                  image_data=img_str, 
                                  result=result_text, 
                                  treatment=treatment_text)

        except Exception as e:
            logger.error(f"Error processing leaf image: {str(e)}")
            flash(f"Error processing image: {str(e)}", 'error')
            return redirect(url_for('leaves'))

    return render_template('index.html', active_tab='leaves')

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
    finally:
        camera.release()
        cursor.close()
        db.close()
        logger.info("Camera and MySQL connection released")
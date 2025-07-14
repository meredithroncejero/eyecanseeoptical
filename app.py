import os
import pickle
import urllib.request
from datetime import datetime, date, time
from threading import Timer

import cv2
import mediapipe as mp
import psycopg2
import pytz
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, render_template, json
from flask import Response
from flask import redirect, url_for
from flask import request, flash
from psycopg2.extras import RealDictCursor
from torchvision import models, transforms
from werkzeug.security import generate_password_hash

from ml_utils import extract_features

app = Flask(__name__)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Load Haar cascade
conn = psycopg2.connect(
    "postgresql://postgres.mggobpvspdsuimokmlwc:EyEcansEEoptical@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres")
cursor = conn.cursor(cursor_factory=RealDictCursor)

url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
urllib.request.urlretrieve(url, "haarcascade_frontalface_default.xml")

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
# ✅ Set a unique and secret key
app.secret_key = os.environ.get('SECRET_KEY', 'fallback_dev_key')

camera = None
camera_timer = None
camera_active = False

EYEWEAR_CLASSES = ['cat_eye', 'geometric', 'half_rim', 'oval', 'rectangle', 'round']

best_match_map = {
    "oval": ["rectangle", "square", "cat_eye", "geometric", "wayfarer"],
    "round": ["rectangle", "square", "browline", "wayfarer"],
    "square": ["round", "oval", "cat_eye"],
    "heart": ["oval", "cat_eye", "round"],
    "diamond": ["oval", "rimless", "round"],
    "oblong": ["tall", "full_rim", "aviator"],
}


def load_eyewear_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(EYEWEAR_CLASSES))
    model.load_state_dict(torch.load("eyewear_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model


eyewear_model = load_eyewear_model()


def predict_eyewear_style(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = eyewear_model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        return EYEWEAR_CLASSES[predicted.item()]


# Load the model
with open("face_shape_model.pkl", "rb") as f:
    model = pickle.load(f)

# Get 3 most recent images in static/Photos
photo_dir = "static/Photos"
image_paths = sorted(
    [os.path.join(photo_dir, f) for f in os.listdir(photo_dir) if f.endswith(".jpg")],
    key=os.path.getmtime,
    reverse=True
)[:3]

# Predict face shape for each image
for path in image_paths:
    print(f"[INFO] Predicting for {path}")
    features = extract_features(path)
    if features:
        shape = model.predict([features])[0]
        print(f"[✅] Predicted face shape: {shape}")
    else:
        print(f"[❌] Could not extract features from: {path}")


def start_camera():
    global camera, camera_active
    if not camera_active:
        print("[CAMERA STATUS] ✅ Camera is ON")
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        camera_active = True


def stop_camera():
    global camera, camera_active
    if camera_active:
        camera.release()
        camera = None
        camera_active = False
        print("[CAMERA STATUS] 🔌 Camera is OFF")


def reset_camera_timer():
    global camera_timer
    if camera_timer:
        camera_timer.cancel()
    camera_timer = Timer(2.0, stop_camera)  # 2 seconds after last access
    camera_timer.start()


@app.route('/')
def index():
    tz = pytz.UTC  # Change this to your DB timezone if different

    today = date.today()
    today_start = tz.localize(datetime.combine(today, time.min))
    today_end = tz.localize(datetime.combine(today, time.max))

    print("today_start:", today_start)
    print("today_end:", today_end)

    # Count patients added today (assuming patient.date is a date field)
    cursor.execute("SELECT COUNT(*) FROM patient WHERE date = %s", (today,))
    patients_today = cursor.fetchone()['count']

    # Count appointments scheduled today between start and end datetime
    cursor.execute("""
        SELECT COUNT(*) FROM appointment
        WHERE appointment_date >= %s AND appointment_date <= %s
    """, (today_start, today_end))
    appointments_today = cursor.fetchone()['count']

    # Count pending payments for today
    cursor.execute("""
        SELECT COUNT(*) FROM invoice
        WHERE transaction_date::date = %s
        AND NOT EXISTS (
            SELECT 1 FROM unnest(payment) AS p
            WHERE p->>'payment_status' = 'Paid'
        )
    """, (today,))
    pending_payments = cursor.fetchone()['count']

    # Count paid invoices based on payment.date_paid matching today
    cursor.execute("""
        SELECT COUNT(*) FROM invoice
        WHERE EXISTS (
            SELECT 1 FROM unnest(payment) AS p
            WHERE p->>'payment_status' = 'Paid'
            AND (p->>'date_paid')::date = %s
        )
    """, (today,))
    sales_today = cursor.fetchone()['count']

    # Recent appointments before today_start
    cursor.execute("""
        SELECT appointment_id, patient_fname, patient_minitial, patient_lname,
               purpose, appointment_date, status
        FROM appointment
        WHERE appointment_date < %s
        ORDER BY appointment_date DESC
        LIMIT 10
    """, (today_start,))
    recent_appointments = cursor.fetchall()

    # Upcoming appointments from today_start onwards
    cursor.execute("""
        SELECT appointment_id, patient_fname, patient_minitial, patient_lname,
               purpose, appointment_date, status
        FROM appointment
        WHERE appointment_date >= %s
        ORDER BY appointment_date ASC
        LIMIT 10
    """, (today_start,))
    upcoming_appointments = cursor.fetchall()

    print("Recent appointments:", recent_appointments)
    print("Upcoming appointments:", upcoming_appointments)

    # Monthly revenue from invoice table
    cursor.execute("""
        SELECT 
            DATE_TRUNC('month', transaction_date) AS month,
            SUM(CAST(total_price AS numeric)) AS total
        FROM invoice
        GROUP BY month
        ORDER BY month ASC
        LIMIT 12
    """)
    monthly_data = cursor.fetchall()

    month_labels = [row['month'].strftime('%b %Y') for row in monthly_data]
    monthly_totals = [float(row['total']) for row in monthly_data]

    target = 10000
    total_actual = sum(monthly_totals)
    sales_target = target * len(monthly_totals)

    return render_template('index.html',
                           dashboard={
                               'patients': patients_today,
                               'appointments': appointments_today,
                               'pending_payments': pending_payments,
                               'sales': sales_today
                           },
                           recent_appointments=recent_appointments,
                           upcoming_appointments=upcoming_appointments,
                           month_labels=month_labels,
                           monthly_totals=monthly_totals,
                           sales_target=sales_target,
                           sales_actual=total_actual
                           )


# Try using Pi Camera, fallback to USB/laptop cam
try:
    camera = cv2.VideoCapture(0)  # default camera (USB/laptop)
    if not camera.isOpened():
        raise Exception("Default camera not available")
except:
    camera = None


def gen_frames():
    global camera, camera_active
    start_camera()

    if camera is None or not camera.isOpened():
        print("[CAMERA STATUS] ❌ Camera not available.")
        return

    if not camera_active:
        print("[CAMERA STATUS] ✅ Camera is ON")
        camera_active = True

    while True:
        reset_camera_timer()
        success, frame = camera.read()
        if not success:
            print("[ERROR] Failed to read from camera.")
            break

        frame = cv2.flip(frame, 1)

        frame_height, frame_width = frame.shape[:2]
        center_x, center_y = frame_width // 2, frame_height // 2
        axis_x, axis_y = 110, 140  # Radius of ellipse (horizontal, vertical)

        # Convert BGR to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        alignment_ok = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Use nose tip (landmark index 1)
                nose = face_landmarks.landmark[1]
                nose_x = int(nose.x * frame.shape[1])
                nose_y = int(nose.y * frame.shape[0])

                # Draw all landmarks
                for pt in face_landmarks.landmark:
                    x = int(pt.x * frame.shape[1])
                    y = int(pt.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # Check if the nose tip is within the ellipse
                ellipse_eq = ((nose_x - center_x) ** 2) / (axis_x ** 2) + ((nose_y - center_y) ** 2) / (axis_y ** 2)
                if ellipse_eq <= 1.0:
                    alignment_ok = True

        # Draw the oval guide
        oval_color = (0, 255, 0) if alignment_ok else (0, 0, 255)
        cv2.ellipse(frame, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, oval_color, 2)

        # Message
        message = "✅ Aligned - You may take a photo" if alignment_ok else "🔴 Please align your face in the oval"
        text_color = (0, 255, 0) if alignment_ok else (0, 0, 255)
        cv2.putText(frame, message, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def crop_eyeglass_region(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the face cascade
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Adjust the region to likely eyeglass area
        eyeglass_y = y + int(h * 0.25)
        eyeglass_h = int(h * 0.3)
        eyeglass_x = x
        eyeglass_w = w

        cropped = img[eyeglass_y:eyeglass_y + eyeglass_h, eyeglass_x:eyeglass_x + eyeglass_w]

        # Save to static/Cropped/
        cropped_dir = os.path.join("static", "Cropped")
        os.makedirs(cropped_dir, exist_ok=True)

        base = os.path.basename(image_path)
        name, ext = os.path.splitext(base)
        save_path = os.path.join(cropped_dir, f"cropped_{name}.jpg")
        cv2.imwrite(save_path, cropped)

        print(f"[✅] Cropped eyeglass region saved to {save_path}")
        return  # Only one face handled for now


@app.route('/choose-frame')
def choose_frame():
    photo_dir = os.path.join('static', 'Photos')
    if not os.path.exists(photo_dir):
        os.makedirs(photo_dir)

    # List up to 3 recent photos
    image_paths = sorted(
        ['Photos/' + f for f in os.listdir(photo_dir) if f.endswith('.jpg')],
        key=lambda x: os.path.getmtime(os.path.join('static', x)),
        reverse=True
    )[:3]

    return render_template('choose_frame.html', image_paths=image_paths)


@app.route('/open-camera')
def open_camera():
    return render_template('choose_frame.html')  # Make sure this file exists


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/take_photo')
def take_photo():
    if camera is None:
        return "Camera not available", 500

    ret, frame = camera.read()
    if not ret:
        return "Failed to capture image", 500

    # ✅ Flip the frame horizontally to match live preview
    frame = cv2.flip(frame, 1)

    # Save photo
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"static/Photos/photo_{timestamp}.jpg"
    cv2.imwrite(filename, frame)

    # ✅ Crop the eyeglass region right after saving
    crop_eyeglass_region(filename)

    # Cleanup older images (keep only 3)
    photo_dir = os.path.join('static', 'Photos')
    photos = sorted(
        [f for f in os.listdir(photo_dir) if f.endswith('.jpg')],
        key=lambda x: os.path.getmtime(os.path.join(photo_dir, x)),
        reverse=True
    )

    for extra_photo in photos[3:]:
        os.remove(os.path.join(photo_dir, extra_photo))

    return redirect(url_for('choose_frame'))


def detect_face_shape(image_path):
    features = extract_features(image_path)
    if features:
        prediction = model.predict([features])[0]
        return prediction
    else:
        return "Unknown"


@app.route('/analyze', methods=['POST'])
def analyze():
    from ml_utils import extract_features
    import pickle
    import numpy as np

    with open("face_shape_model.pkl", "rb") as f:
        model = pickle.load(f)

    photo_dir = os.path.join('static', 'Photos')
    image_paths = sorted(
        [os.path.join(photo_dir, f) for f in os.listdir(photo_dir) if f.endswith('.jpg')],
        key=os.path.getmtime,
        reverse=True
    )[:3]

    feature_list = []
    valid_images = []
    frame_predictions = []

    for path in image_paths:
        features = extract_features(path)
        if not features:
            continue

        feature_list.append(features)
        valid_images.append(path)

        crop_eyeglass_region(path)  # saves cropped version

        name = os.path.splitext(os.path.basename(path))[0]
        cropped_path = os.path.join('static', 'Cropped', f"cropped_{name}.jpg")

        if os.path.exists(cropped_path):
            frame = predict_eyewear_style(cropped_path)
        else:
            frame = "Unknown"

        frame_predictions.append((os.path.basename(path), frame))

    if not feature_list:
        return "Failed to extract features", 400

    avg_features = np.mean(feature_list, axis=0)
    predicted_shape = model.predict([avg_features])[0]

    # Pick the best matching uploaded image based on frame recommendation
    recommended_frames = best_match_map.get(predicted_shape.lower(), [])

    best_match = None
    for img, frame in frame_predictions:
        if frame.lower() in recommended_frames:
            best_match = (img, frame)
            break

    # If no good match found, just pick the first image as fallback
    if best_match is None and frame_predictions:
        best_match = frame_predictions[0]

    return render_template("results.html",
                           uploaded_images=[os.path.basename(p) for p in valid_images],
                           average_features=[round(f, 4) for f in avg_features],
                           predicted_shape=predicted_shape,
                           frame_predictions=frame_predictions)


@app.template_filter('format_time')
def format_time(value):
    return value.strftime('%I:%M %p') if isinstance(value, datetime) else value


@app.route('/appointments')
def appointments():
    cursor.execute("SELECT * FROM appointment")
    appointments = cursor.fetchall()
    return render_template("appointment.html", appointments=appointments)


@app.route('/filter_appointments')
def filter_appointments():
    status = request.args.get('status')
    appt_date = request.args.get('appointment_date')
    query = "SELECT * FROM appointment WHERE 1=1"
    params = []

    if status and status.lower() != "all":
        query += " AND status = %s"
        params.append(status)
    if appt_date:
        query += " AND appointment_date = %s"
        params.append(appt_date)

    cursor.execute(query, tuple(params))
    appointments = cursor.fetchall()
    return render_template("appointment.html", appointments=appointments)


@app.route('/update_appointment_status/<int:appointment_id>', methods=['POST'])
def update_appointment_status(appointment_id):
    new_status = request.form.get('status')
    if new_status:
        cursor.execute("UPDATE appointment SET status = %s WHERE appointment_id = %s", (new_status, appointment_id))
        conn.commit()
    return redirect(request.referrer or url_for('appointments'))  # or your appointments route


@app.route('/delete_photo', methods=['POST'])
def delete_photo():
    filename = request.form.get('filename')
    if filename:
        file_path = os.path.join(app.static_folder, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    return redirect(url_for('choose_frame'))


@app.route('/patients')
def patient_records():
    status = request.args.get('status', 'All')

    # Update number_of_visit for each patient based on count of invoices
    cursor.execute("""
        UPDATE patient SET number_of_visit = sub.invoice_count
        FROM (
            SELECT patient_id, COUNT(*) AS invoice_count
            FROM invoice
            GROUP BY patient_id
        ) AS sub
        WHERE patient.patient_id = sub.patient_id
    """)
    conn.commit()

    # Fetch patients according to filter
    if status == 'All':
        cursor.execute("SELECT * FROM patient")
    else:
        cursor.execute("SELECT * FROM patient WHERE status = %s", (status,))

    patients = cursor.fetchall()
    return render_template("tables.html", patients=patients, status=status)


@app.route('/add_patient', methods=['POST'])
def add_patient():
    # Extract form data
    patient_fname = request.form.get('patient_fname')
    patient_minitial = request.form.get('patient_minitial')
    patient_lname = request.form.get('patient_lname')
    email = request.form.get('email')
    age = request.form.get('age')
    birthday = request.form.get('birthday')  # YYYY-MM-DD format string
    gender = request.form.get('gender')
    contact_details = request.form.get('contact_details')
    province = request.form.get('province')
    city = request.form.get('city')
    barangay = request.form.get('barangay')
    street = request.form.get('street')
    occupation = request.form.get('occupation')
    date = request.form.get('date')  # YYYY-MM-DD format string

    # Validation (optional)
    if not patient_fname or not patient_lname:
        flash('First and last name are required.', 'danger')
        return redirect(url_for('index'))

    cursor.execute("""
        INSERT INTO patient (
            patient_fname, patient_minitial, patient_lname, email, age, birthday,
            gender, contact_details, province, city, barangay, street, occupation, date
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        patient_fname, patient_minitial, patient_lname, email, age, birthday,
        gender, contact_details, province, city, barangay, street, occupation, date
    ))

    # ✅ Commit using the actual connection object
    conn.commit()

    flash('New patient added successfully!', 'success')
    return redirect(url_for('index'))


@app.route('/update_eye_results', methods=['POST'])
def update_eye_results():
    # Get form data
    result_id = request.form.get('eye_results_id')
    medical_history = request.form.get('medical_history')
    old_rx_od = request.form.get('old_rx_od')
    old_rx_os = request.form.get('old_rx_os')
    old_va_od = request.form.get('old_va_od')
    old_va_os = request.form.get('old_va_os')
    old_add_od = request.form.get('old_add_od')
    old_add_os = request.form.get('old_add_os')
    bp = request.form.get('bp')
    ishihara_result = request.form.get('ishihara_result')

    if not result_id:
        flash('Missing result ID. Cannot update.', 'danger')
        return redirect(url_for('patient_records'))  # Or the appropriate route

    try:
        cursor.execute("""
            UPDATE eyeresult
            SET medical_history = %s,
                old_rx_od = %s,
                old_rx_os = %s,
                old_va_od = %s,
                old_va_os = %s,
                old_add_od = %s,
                old_add_os = %s,
                bp = %s,
                ishihara_result = %s
            WHERE eye_results_id = %s
        """, (
            medical_history,
            old_rx_od,
            old_rx_os,
            old_va_od,
            old_va_os,
            old_add_od,
            old_add_os,
            bp,
            ishihara_result,
            result_id
        ))

        conn.commit()
        flash('Eye results updated successfully.', 'success')
    except Exception as e:
        conn.rollback()
        flash(f'Error updating eye results: {str(e)}', 'danger')

    return redirect(url_for('patient_records'))  # Replace 'index' with your view function name


@app.route('/update_prescription', methods=['POST'])
def update_prescription():
    prescription_id = request.form.get('prescription_id')
    prescription_date = request.form.get('prescription_date')
    distance_rx_od = request.form.get('distance_rx_od')
    distance_rx_os = request.form.get('distance_rx_os')
    contact_rx_od = request.form.get('contact_rx_od')
    contact_rx_os = request.form.get('contact_rx_os')
    reading_rx_od = request.form.get('reading_rx_od')
    reading_rx_os = request.form.get('reading_rx_os')
    sph_od = request.form.get('sph_od')
    sph_os = request.form.get('sph_os')
    cyl_od = request.form.get('cyl_od')
    cyl_os = request.form.get('cyl_os')
    axis_od = request.form.get('axis_od')
    axis_os = request.form.get('axis_os')
    va_od = request.form.get('va_od')
    va_os = request.form.get('va_os')
    add_od = request.form.get('add_od')
    add_os = request.form.get('add_os')
    mono_od = request.form.get('mono_od')
    pd_os = request.form.get('pd_os')
    seg_ht_od = request.form.get('seg_ht_od')
    vert_ht_os = request.form.get('vert_ht_os')
    pd = request.form.get('pd')

    if not prescription_id:
        flash('Missing prescription ID. Cannot update.', 'danger')
        return redirect(url_for('patient_records'))  # Adjust this route

    try:
        cursor.execute("""
            UPDATE prescription
            SET prescription_date = %s,
                distance_rx_od = %s,
                distance_rx_os = %s,
                contact_rx_od = %s,
                contact_rx_os = %s,
                reading_rx_od = %s,
                reading_rx_os = %s,
                sph_od = %s,
                sph_os = %s,
                cyl_od = %s,
                cyl_os = %s,
                axis_od = %s,
                axis_os = %s,
                va_od = %s,
                va_os = %s,
                add_od = %s,
                add_os = %s,
                mono_od = %s,
                pd_os = %s,
                seg_ht_od = %s,
                vert_ht_os = %s,
                pd = %s
            WHERE prescription_id = %s
        """, (
            prescription_date,
            distance_rx_od,
            distance_rx_os,
            contact_rx_od,
            contact_rx_os,
            reading_rx_od,
            reading_rx_os,
            sph_od,
            sph_os,
            cyl_od,
            cyl_os,
            axis_od,
            axis_os,
            va_od,
            va_os,
            add_od,
            add_os,
            mono_od,
            pd_os,
            seg_ht_od,
            vert_ht_os,
            pd,
            prescription_id
        ))

        conn.commit()
        flash('Prescription updated successfully.', 'success')
    except Exception as e:
        conn.rollback()
        flash(f'Error updating prescription: {str(e)}', 'danger')

    return redirect(url_for('patient_records'))  # Adjust to your actual route


@app.route('/invoices')
def invoices():
    cursor.execute("SELECT * FROM invoice")
    invoices = cursor.fetchall()

    total_earnings = 0
    overdue_count = 0
    today = date.today()

    for inv in invoices:
        total_earnings += inv['total_price'] or 0

        payments = inv.get('payment') or []
        if payments:
            latest_payment = payments[-1]
            inv['payment_method'] = latest_payment.get('payment_method')
            inv['payment_status'] = latest_payment.get('payment_status')
        else:
            inv['payment_method'] = None
            inv['payment_status'] = None

        if inv.get('payment_status') == 'Partial':
            overdue_count += 1

    # Get most frequent patient
    cursor.execute("""
        SELECT * FROM patient
        ORDER BY number_of_visit DESC
        LIMIT 1
    """)
    most_frequent_patient = cursor.fetchone()

    # 🔧 Get all patients (for the dropdown list in the modal)
    cursor.execute("SELECT patient_id, patient_fname, patient_minitial, patient_lname FROM patient")
    all_patients = cursor.fetchall()

    # 🔄 Convert each row to dictionary (in case it's not already)
    all_patients_json = []
    for p in all_patients:
        all_patients_json.append({
            "patient_id": p['patient_id'],
            "patient_fname": p['patient_fname'],
            "patient_minitial": p['patient_minitial'] or "",
            "patient_lname": p['patient_lname']
        })

    return render_template(
        'invoice.html',
        invoices=invoices,
        total_earnings=total_earnings,
        overdue_count=overdue_count,
        most_frequent_patient=most_frequent_patient,
        all_patients=all_patients_json  # ✅ Now available for tojson
    )


@app.route('/update_invoice', methods=['POST'])
def update_invoice():
    invoice_id = request.form['invoice_id']
    invoice_number = request.form['invoice_number']
    transaction_date = request.form['transaction_date']
    claim_date = request.form['claim_date']
    frame_price = request.form['frame_price']
    lens_price = request.form['lens_price']
    additional_price = request.form['additional_price']
    total_price = request.form['total_price']
    deposit_amount = request.form['deposit_amount']
    balance_due = request.form['balance_due']

    try:
        cursor.execute("""
            UPDATE invoice
            SET invoice_number = %s,
                transaction_date = %s,
                claim_date = %s,
                frame_price = %s,
                lens_price = %s,
                additional_price = %s,
                total_price = %s,
                deposit_amount = %s,
                balance_due = %s
            WHERE invoice_id = %s
        """, (
            invoice_number, transaction_date, claim_date,
            frame_price, lens_price, additional_price,
            total_price, deposit_amount, balance_due, invoice_id
        ))
        conn.commit()
        flash('Invoice updated successfully.', 'success')
    except Exception as e:
        conn.rollback()
        flash(f'Error updating invoice: {str(e)}', 'danger')

    return redirect(url_for('patient_records'))


@app.route('/add_payment', methods=['POST'])
def add_payment():
    invoice_id = request.form['invoice_id']
    date_paid = request.form['date_paid']
    amount_paid = request.form['amount_paid']
    payment_method = request.form['payment_method']
    payment_status = request.form['payment_status']

    # Generate 'time_ago'
    today = datetime.today()
    paid_date = datetime.strptime(date_paid, '%Y-%m-%d')
    days_ago = (today - paid_date).days
    time_ago = f"{days_ago} day(s) ago" if days_ago > 0 else "Today"

    # Stringify the payment as a single JSON string to store in TEXT[]
    payment_entry = json.dumps({
        "date_paid": date_paid,
        "amount_paid": amount_paid,
        "payment_method": payment_method,
        "payment_status": payment_status,
        "time_ago": time_ago
    })

    try:
        # Append to PostgreSQL array using array_append
        cursor.execute("""
            UPDATE invoice
            SET payment = array_append(payment, %s)
            WHERE invoice_id = %s
        """, (payment_entry, invoice_id))
        conn.commit()
        flash('Payment added successfully.', 'success')
    except Exception as e:
        conn.rollback()
        flash(f'Error adding payment: {str(e)}', 'danger')

    return redirect(url_for('patient_records'))


@app.route('/add_invoice', methods=['POST'])
def add_invoice():
    try:
        patient_type = request.form.get('patient_type')  # 'new' or 'returning'
        patient_id = request.form.get('patient_id')

        if not patient_id:
            flash('Patient ID is required.', 'danger')
            return redirect(url_for('invoices'))

        if patient_type == 'new':
            patient_fname = request.form['patient_fname']
            patient_minitial = request.form.get('patient_minitial', '')
            patient_lname = request.form['patient_lname']

            cursor.execute("""
                INSERT INTO patient (patient_id, patient_fname, patient_minitial, patient_lname, number_of_visit)
                VALUES (%s, %s, %s, %s, 1)
            """, (patient_id, patient_fname, patient_minitial, patient_lname))
        elif patient_type == 'returning':
            cursor.execute("""
                UPDATE patient SET number_of_visit = number_of_visit + 1 WHERE patient_id = %s
            """, (patient_id,))
        else:
            flash('Invalid patient type.', 'danger')
            return redirect(url_for('invoices'))

        # Extract invoice data
        invoice_number = request.form['invoice_number']
        transaction_date = request.form['transaction_date']
        claim_date = request.form['claim_date']
        frame_price = float(request.form['frame_price'])
        frame_price = float(request.form['frame_price'])
        lens_price = float(request.form['lens_price'])
        additional_price = float(request.form['additional_price'])
        total_price = float(request.form['total_price'])
        deposit_amount = float(request.form['deposit_amount'])
        balance_due = float(request.form['balance_due'])

        cursor.execute("""
            INSERT INTO invoice (
                patient_id, invoice_number, transaction_date, claim_date,
                frame_price, lens_price, additional_price, total_price,
                deposit_amount, balance_due, payment,
                patient_fname, patient_minitial, patient_lname
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            patient_id, invoice_number, transaction_date, claim_date,
            frame_price, lens_price, additional_price, total_price,
            deposit_amount, balance_due, [],
            request.form['patient_fname'],
            request.form.get('patient_minitial', ''),
            request.form['patient_lname']
        ))

        conn.commit()
        flash('Invoice added successfully.', 'success')

    except Exception as e:
        conn.rollback()
        flash(f'Error adding invoice: {e}', 'danger')

    return redirect(url_for('invoices'))


@app.route('/delete_invoice/<int:invoice_id>', methods=['POST'])
def delete_invoice(invoice_id):
    try:
        cursor.execute("DELETE FROM invoice WHERE invoice_id = %s", (invoice_id,))
        conn.commit()
        flash('Invoice deleted successfully.', 'success')
    except Exception as e:
        conn.rollback()
        flash(f'Error deleting invoice: {e}', 'danger')
    return redirect(url_for('invoices'))  # or wherever you want to redirect after deletion


@app.route('/patient/<patient_id>')
def patient_details(patient_id):
    cursor.execute("SELECT * FROM patient WHERE patient_id = %s", (patient_id,))
    patient = cursor.fetchone()
    if not patient:
        return "Patient not found", 404

    cursor.execute("SELECT * FROM eyeresult WHERE patient_id = %s", (patient_id,))
    eye_results = cursor.fetchall()

    cursor.execute("SELECT * FROM prescription WHERE patient_id = %s", (patient_id,))
    prescriptions = cursor.fetchall()

    cursor.execute("SELECT * FROM invoice WHERE patient_id = %s", (patient_id,))
    invoices = cursor.fetchall()

    return render_template("patient details.html",
                           patient=patient,
                           eye_results=eye_results,
                           prescriptions=prescriptions,
                           invoices=invoices)


@app.route('/patient/<patient_id>/history')
def patient_history(patient_id):
    cursor.execute("SELECT * FROM patient WHERE patient_id = %s", (patient_id,))
    patient = cursor.fetchone()
    if not patient:
        return "Patient not found", 404

    cursor.execute("SELECT * FROM appointment WHERE patient_id = %s", (patient_id,))
    appointments = cursor.fetchall()

    cursor.execute("SELECT * FROM prescription WHERE patient_id = %s", (patient_id,))
    prescriptions = cursor.fetchall()

    presc_lookup = {
        p['prescription_date']: {
            'Eye_Exam_Results': p.get('va_od', 'N/A'),
            'Vision_Prescription': f"{p.get('sph_od', '')}/{p.get('sph_os', '')}"
        }
        for p in prescriptions
    }

    history_data = []
    for appt in appointments:
        date_str = appt['appointment_date']
        history_data.append({
            'appointment_id': appt.get('appointment_id', 'N/A'),
            'appointment_date': date_str,
            'purpose': appt.get('purpose', 'N/A'),
            'status': appt.get('status', 'N/A'),
        })

    return render_template("patient history.html", patient=patient, history_data=history_data)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_type = request.form['user_type']
        first_name = request.form['first_name']
        middle_initial = request.form['middle_initial']
        last_name = request.form['last_name']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)

        print("Form received:", request.form.to_dict())

        try:
            if user_type == 'admin':
                role = request.form.get('role')
                contact_info = request.form.get('contact_info')

                print("Inserting admin record...")
                cursor.execute("""
                    INSERT INTO admin (
                        admin_fname, admin_minitial, admin_lname,
                        admin_username, email, password, role, contact_info
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    first_name, middle_initial, last_name,
                    username, email, hashed_password, role, contact_info
                ))
            else:
                print("Inserting user record...")
                cursor.execute("""
                    INSERT INTO users (
                        user_fname, user_minitial, user_lname,
                        user_username, email, password
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    first_name, middle_initial, last_name,
                    username, email, hashed_password
                ))

            conn.commit()
            flash('Account created successfully!', 'success')
            return redirect(url_for('index'))

        except psycopg2.Error as e:
            conn.rollback()
            print("Database error:", e.pgerror)
            flash(f"Database error: {e.pgerror}", 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Create Flask app with custom template folder
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'app', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'app', 'static'))
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/bp_risk_assessment'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load the trained model and scaler
model = joblib.load('model/ensemble_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))  # Increased length to store full hash
    assessments = db.relationship('Assessment', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password, method='scrypt')

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Assessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    # Personal Information
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.Boolean, nullable=False)  # True for Male, False for Female
    
    # Blood Pressure and Heart
    systolic = db.Column(db.Float, nullable=False)
    diastolic = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)
    bp_medication = db.Column(db.Boolean, nullable=False)  # True for Yes, False for No
    
    # Health Metrics
    bmi = db.Column(db.Float, nullable=False)
    total_cholesterol = db.Column(db.Float, nullable=False)
    
    # Medical Conditions
    diabetes = db.Column(db.Boolean, nullable=False)  # True for Yes, False for No
    blood_glucose = db.Column(db.Float, nullable=False)
    smoking_status = db.Column(db.Boolean, nullable=False)  # True for Current Smoker, False for Non-smoker
    
    risk_status = db.Column(db.String(20), nullable=False)
    recommendations = db.Column(db.Text, nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        print(f"Login attempt for user: {username}")
        print(f"Login password length: {len(password)}")  # Debug password length
        
        user = User.query.filter_by(username=username).first()
        if user:
            print(f"User found: {username}")
            print(f"Stored password hash: {user.password_hash}")  # Debug stored hash
            if user.check_password(password):
                print(f"Password check passed for: {username}")
                login_user(user)
                return redirect(url_for('dashboard'))
            else:
                print(f"Password check failed for: {username}")
                print(f"Entered password: {password}")  # Debug entered password
        else:
            print(f"User not found: {username}")
            
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        print(f"Attempting to create user: {username}")
        print(f"Password length: {len(password)}")  # Debug password length
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))
        
        user = User(username=username, email=email)
        user.set_password(password)
        print(f"Password hash: {user.password_hash}")  # Debug password hash
        
        db.session.add(user)
        try:
            db.session.commit()
            print(f"Successfully created user: {username}")
        except Exception as e:
            print(f"Error creating user: {str(e)}")
            db.session.rollback()
            flash('An error occurred while creating your account')
            return redirect(url_for('signup'))
        
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/dashboard')
@login_required
def dashboard():
    assessments = Assessment.query.filter_by(user_id=current_user.id).order_by(Assessment.date.desc()).all()
    # Convert assessments to dictionaries for JSON serialization
    assessments_data = [{
        'id': assessment.id,
        'date': assessment.date.strftime('%Y-%m-%d'),
        'systolic': assessment.systolic,
        'diastolic': assessment.diastolic,
        'heart_rate': assessment.heart_rate,
        'risk_status': assessment.risk_status,
        'recommendations': assessment.recommendations
    } for assessment in assessments]
    return render_template('dashboard.html', assessments=assessments, assessments_data=assessments_data)

@app.route('/assessment', methods=['GET', 'POST'])
@login_required
def assessment():
    if request.method == 'POST':
        try:
            # Get form data with validation
            # Personal Information
            age = request.form.get('age')
            gender = request.form.get('gender')  # 1 for Male, 0 for Female
            
            # Blood Pressure and Heart
            systolic = request.form.get('systolic')
            diastolic = request.form.get('diastolic')
            heart_rate = request.form.get('heart_rate')
            bp_medication = request.form.get('bp_medication')  # 1 for Yes, 0 for No
            
            # Health Metrics
            bmi = request.form.get('bmi')
            total_cholesterol = request.form.get('total_cholesterol')
            
            # Medical Conditions
            diabetes = request.form.get('diabetes')  # 1 for Yes, 0 for No
            blood_glucose = request.form.get('blood_glucose')
            smoking_status = request.form.get('smoking_status')  # 1 for Current Smoker, 0 for Non-smoker
            cigs_per_day = request.form.get('cigs_per_day')

            # Validate that all required fields are present
            required_fields = {
                'age': age, 'gender': gender, 'systolic': systolic, 'diastolic': diastolic,
                'heart_rate': heart_rate, 'bp_medication': bp_medication, 'bmi': bmi,
                'total_cholesterol': total_cholesterol, 'diabetes': diabetes,
                'blood_glucose': blood_glucose, 'smoking_status': smoking_status,
                'cigs_per_day': cigs_per_day
            }
            
            if not all(required_fields.values()):
                flash('Please fill in all required fields')
                return redirect(url_for('assessment'))

            # Convert to appropriate types with error handling
            try:
                # Personal Information
                age = int(age)
                gender = bool(int(gender))
                
                # Blood Pressure and Heart
                systolic = float(systolic)
                diastolic = float(diastolic)
                heart_rate = float(heart_rate)
                bp_medication = bool(int(bp_medication))
                
                # Health Metrics
                bmi = float(bmi)
                total_cholesterol = float(total_cholesterol)
                
                # Medical Conditions
                diabetes = bool(int(diabetes))
                blood_glucose = float(blood_glucose)
                smoking_status = bool(int(smoking_status))
                cigs_per_day = float(cigs_per_day)
            except ValueError:
                flash('Please enter valid numbers for all fields')
                return redirect(url_for('assessment'))

            # Validate ranges
            if not (18 <= age <= 120):
                flash('Age must be between 18 and 120 years')
                return redirect(url_for('assessment'))
            if not (70 <= systolic <= 200):
                flash('Systolic pressure must be between 70 and 200 mmHg')
                return redirect(url_for('assessment'))
            if not (40 <= diastolic <= 130):
                flash('Diastolic pressure must be between 40 and 130 mmHg')
                return redirect(url_for('assessment'))
            if not (40 <= heart_rate <= 200):
                flash('Heart rate must be between 40 and 200 bpm')
                return redirect(url_for('assessment'))
            if not (10 <= bmi <= 50):
                flash('BMI must be between 10 and 50')
                return redirect(url_for('assessment'))
            if not (100 <= total_cholesterol <= 500):
                flash('Total cholesterol must be between 100 and 500 mg/dL')
                return redirect(url_for('assessment'))
            if not (50 <= blood_glucose <= 400):
                flash('Blood glucose must be between 50 and 400 mg/dL')
                return redirect(url_for('assessment'))
            if not (0 <= cigs_per_day <= 100):
                flash('Cigarettes per day must be between 0 and 100')
                return redirect(url_for('assessment'))
            
            # Prepare input for model with exact feature names from training
            input_data = pd.DataFrame({
                'male': [int(gender)],
                'age': [age],
                'currentSmoker': [int(smoking_status)],
                'cigsPerDay': [cigs_per_day],
                'BPMeds': [int(bp_medication)],
                'diabetes': [int(diabetes)],
                'totChol': [total_cholesterol],
                'sysBP': [systolic],
                'diaBP': [diastolic],
                'BMI': [bmi],
                'heartRate': [heart_rate],
                'glucose': [blood_glucose]
            }, columns=[
                'male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds',
                'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI',
                'heartRate', 'glucose'
            ])
            
            # Scale the input data
            input_scaled = scaler.transform(input_data)
            
            # Get prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            risk_status = "High Risk" if prediction == 1 else "Low Risk"
            
            # Generate recommendations based on risk status and probabilities
            recommendations = generate_recommendations(
                risk_status, systolic, diastolic, bmi, prediction_proba,
                bp_medication, diabetes, smoking_status, total_cholesterol, blood_glucose,
                cigs_per_day
            )
            
            # Save assessment with all fields
            assessment = Assessment(
                user_id=current_user.id,
                age=age,
                gender=gender,
                systolic=systolic,
                diastolic=diastolic,
                heart_rate=heart_rate,
                bp_medication=bp_medication,
                bmi=bmi,
                total_cholesterol=total_cholesterol,
                diabetes=diabetes,
                blood_glucose=blood_glucose,
                smoking_status=smoking_status,
                risk_status=risk_status,
                recommendations=recommendations
            )
            db.session.add(assessment)
            db.session.commit()
            
            return redirect(url_for('results', assessment_id=assessment.id))
            
        except Exception as e:
            print(f"Error processing assessment: {str(e)}")
            flash('An error occurred while processing your assessment. Please try again.')
            return redirect(url_for('assessment'))
    
    return render_template('assessment.html')

@app.route('/results/<int:assessment_id>')
@login_required
def results(assessment_id):
    assessment = Assessment.query.get_or_404(assessment_id)
    if assessment.user_id != current_user.id:
        return redirect(url_for('dashboard'))
    return render_template('results.html', assessment=assessment)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

def generate_recommendations(risk_status, systolic, diastolic, bmi, prediction_proba,
                          bp_medication, diabetes, smoking_status, total_cholesterol, blood_glucose,
                          cigs_per_day):
    recommendations = []
    
    # General recommendations based on risk status
    if risk_status == "High Risk":
        recommendations.append("⚠️ Important: Schedule an appointment with your healthcare provider immediately.")
        recommendations.append("📊 Monitor your blood pressure daily and keep a log of readings.")
        recommendations.append("🍽️ Reduce sodium intake to less than 1500mg per day.")
        recommendations.append("🏃 Exercise regularly (30 minutes of moderate activity 5 times per week).")
        recommendations.append("🧘 Manage stress through meditation or relaxation techniques.")
        recommendations.append("💊 Take prescribed medications as directed by your healthcare provider.")
    else:
        recommendations.append("✅ Continue monitoring your blood pressure regularly.")
        recommendations.append("🥗 Maintain a healthy diet rich in fruits and vegetables.")
        recommendations.append("🏃 Stay physically active with regular exercise.")
        recommendations.append("🧘 Keep stress levels in check through relaxation techniques.")
        recommendations.append("📅 Schedule regular check-ups with your healthcare provider.")
    
    # Specific recommendations based on measurements and conditions
    if systolic > 140 or diastolic > 90:
        recommendations.append("⚠️ Your blood pressure readings are elevated. Consider lifestyle changes and consult a healthcare provider.")
        recommendations.append("🍽️ Focus on the DASH diet (Dietary Approaches to Stop Hypertension).")
    elif systolic < 90 or diastolic < 60:
        recommendations.append("⚠️ Your blood pressure readings are low. Monitor for symptoms and consult a healthcare provider if concerned.")
        recommendations.append("💧 Stay well-hydrated and consider increasing salt intake if advised by your doctor.")
    
    if bmi > 25:
        recommendations.append("⚖️ Your BMI indicates overweight. Work on maintaining a healthy weight through diet and exercise.")
        recommendations.append("🍽️ Consider consulting a registered dietitian for personalized dietary advice.")
    elif bmi < 18.5:
        recommendations.append("⚖️ Your BMI indicates underweight. Consider consulting a healthcare provider about healthy weight gain strategies.")
    
    if diabetes:
        recommendations.append("🩸 Maintain strict blood glucose control through diet, exercise, and medication as prescribed.")
        recommendations.append("👣 Check your feet daily for any signs of complications.")
    
    if smoking_status:
        recommendations.append("🚭 Smoking significantly increases cardiovascular risk. Consider smoking cessation programs.")
        recommendations.append("💪 Seek support for quitting smoking through healthcare providers or support groups.")
    
    if total_cholesterol > 200:
        recommendations.append("🫀 Your cholesterol is elevated. Focus on heart-healthy foods and regular exercise.")
        recommendations.append("🥑 Include omega-3 rich foods in your diet (fish, nuts, seeds).")
    
    if blood_glucose > 126:
        recommendations.append("🩸 Your blood glucose is elevated. Monitor regularly and follow your diabetes management plan.")
        recommendations.append("🍽️ Limit refined carbohydrates and sugary foods.")
    
    # Add lifestyle recommendations
    recommendations.append("🌙 Aim for 7-9 hours of quality sleep each night.")
    recommendations.append("💧 Stay hydrated with 8-10 glasses of water daily.")
    recommendations.append("🧘 Practice stress management techniques regularly.")
    
    return "\n".join(recommendations)

if __name__ == '__main__':
    with app.app_context():
        try:
            print("Creating database tables...")
            db.create_all()
            print("Database tables created successfully!")
        except Exception as e:
            print(f"Error creating database tables: {str(e)}")
    app.run(debug=True) 
# BP GUARD - Blood Pressure Risk Assessment System

## Overview
BP GUARD is an AI-powered web application designed to help users monitor and assess their blood pressure health risks. The system combines machine learning models with a user-friendly interface to provide personalized risk assessments and health recommendations.

## Features
- User authentication and account management
- Blood pressure assessment form with comprehensive health metrics
- Real-time risk assessment using machine learning models
- Interactive blood pressure trend visualization
- Personalized health recommendations
- Assessment history tracking
- Responsive and modern user interface

## Technical Stack
- **Backend**: Python/Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **Database**: MySQL
- **Machine Learning**: scikit-learn
- **Visualization**: Chart.js
- **Authentication**: Flask-Login
- **Styling**: Bootstrap 5

## Project Structure
```
bp-risk-assesment/
├── app/
│   ├── static/
│   │   ├── css/
│   │   │   ├── bootstrap.min.css
│   │   │   └── style.css
│   │   └── js/
│   │       ├── bootstrap.bundle.min.js
│   │       └── chart.min.js
│   ├── templates/
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── assessment.html
│   │   ├── results.html
│   │   ├── login.html
│   │   └── signup.html
│   └── __init__.py
├── model/
│   ├── ensemble_model.pkl
│   └── scaler.pkl
├── app.py
├── requirements.txt
└── ABOUT_THE_PROJECT.md
```

## Machine Learning Model

### Model Architecture
The system uses an ensemble model trained on blood pressure risk assessment data. The model takes into account various health metrics to predict the risk level of cardiovascular issues.

### Features Used
1. Age
2. Gender (Male/Female)
3. Smoking Status
4. Cigarettes per Day
5. Blood Pressure Medication
6. Diabetes Status
7. Total Cholesterol
8. Systolic Blood Pressure
9. Diastolic Blood Pressure
10. BMI
11. Heart Rate
12. Blood Glucose

### Model Training
- The model was trained using scikit-learn's ensemble methods
- Features were preprocessed using StandardScaler
- The model predicts binary risk classification (High Risk/Low Risk)

### Model Performance
- The model provides risk assessment with probability scores
- Recommendations are generated based on the risk level and contributing factors

## Database Schema

### Users Table
```sql
CREATE TABLE user (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL
);
```

### Assessments Table
```sql
CREATE TABLE assessment (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    date DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    age INT NOT NULL,
    gender BOOLEAN NOT NULL,
    systolic FLOAT NOT NULL,
    diastolic FLOAT NOT NULL,
    heart_rate FLOAT NOT NULL,
    bp_medication BOOLEAN NOT NULL,
    bmi FLOAT NOT NULL,
    total_cholesterol FLOAT NOT NULL,
    diabetes BOOLEAN NOT NULL,
    blood_glucose FLOAT NOT NULL,
    smoking_status BOOLEAN NOT NULL,
    risk_status VARCHAR(20) NOT NULL,
    recommendations TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES user(id)
);
```

## Security Features
- Password hashing using Werkzeug's scrypt algorithm
- Session management with Flask-Login
- Protected routes requiring authentication
- Input validation and sanitization
- Secure password storage

## User Interface Components

### Dashboard
- Welcome section with user profile
- Blood pressure trends chart with filtering options
- Latest assessment summary
- Assessment history table

### Assessment Form
- Comprehensive health metrics input
- Real-time validation
- Clear categorization of health indicators
- User-friendly input controls

### Results Page
- Risk status display
- Detailed assessment information
- Personalized recommendations
- Navigation options

## Future Enhancements
1. Mobile application development
2. Integration with health monitoring devices
3. Export functionality for medical records
4. Advanced analytics and reporting
5. Healthcare provider dashboard
6. Multi-language support
7. Dark mode theme
8. Email notifications for regular check-ups

## Installation and Setup
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up MySQL database:
   ```sql
   CREATE DATABASE bp_risk_assessment;
   ```
5. Configure environment variables:
   - Set `SECRET_KEY` in app.py
   - Update database connection string
6. Run the application:
   ```bash
   python app.py
   ```

## Dependencies
- Flask==2.0.1
- Flask-SQLAlchemy==2.5.1
- Flask-Login==0.5.0
- scikit-learn==0.24.2
- pandas==1.3.3
- numpy==1.21.2
- mysqlclient==2.0.3
- Werkzeug==2.0.1
- joblib==1.0.2

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset providers


## Contacts
johnwahome966@gmail.com
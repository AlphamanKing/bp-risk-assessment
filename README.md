# Blood Pressure Risk Assessment System

An AI-powered web application for assessing blood pressure risk in patients.

## Features

- User authentication (login/signup)
- Dashboard with historical data visualization
- Health assessment form
- AI-powered risk prediction
- Personalized recommendations

## Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript
- Bootstrap 5
- Chart.js

### Backend
- Flask (Python)
- MySQL
- XAMPP

## Setup Instructions

1. Install XAMPP and start MySQL server
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Set up the database:
   - Start XAMPP MySQL server
   - Create a database named 'bp_risk_assessment'
   - Import the schema from `database/schema.sql`

5. Run the application:
   ```
   python app.py
   ```

## Project Structure

```
bp-risk-assessment/
├── app/
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── img/
│   ├── templates/
│   ├── models/
│   ├── routes/
│   └── utils/
├── database/
├── model/
├── venv/
├── app.py
├── config.py
└── requirements.txt
```

## Download Required Frontend Libraries

1. Bootstrap 5: Download from https://getbootstrap.com/docs/5.1/getting-started/download/
   - Place in `app/static/css/bootstrap.min.css`
   - Place in `app/static/js/bootstrap.bundle.min.js`

2. Chart.js: Download from https://cdn.jsdelivr.net/npm/chart.js
   - Place in `app/static/js/chart.min.js` 
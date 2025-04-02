CREATE DATABASE IF NOT EXISTS bp_risk_assessment;
USE bp_risk_assessment;

CREATE TABLE IF NOT EXISTS user (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(128) NOT NULL
);

CREATE TABLE IF NOT EXISTS assessment (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    date DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    systolic FLOAT NOT NULL,
    diastolic FLOAT NOT NULL,
    heart_rate FLOAT NOT NULL,
    age INT NOT NULL,
    bmi FLOAT NOT NULL,
    risk_status VARCHAR(20) NOT NULL,
    recommendations TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES user(id)
); 
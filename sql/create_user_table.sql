CREATE DATABASE IF NOT EXISTS hiresense_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE hiresense_db;

CREATE TABLE IF NOT EXISTS users (
  id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  unique_id CHAR(36) NOT NULL UNIQUE,
  email VARCHAR(255) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  full_name VARCHAR(200) NOT NULL,
  role ENUM('recruiter','admin') NOT NULL DEFAULT 'recruiter',
  recruiter_role VARCHAR(100),
  business_name VARCHAR(255),
  website_url VARCHAR(512),
  linkedin_url VARCHAR(512),
  no_of_employees ENUM('0-25','25-50','50-100','100-300','300+') NULL,
  phone VARCHAR(30),
  status ENUM('pending','approved','rejected','disabled') NOT NULL DEFAULT 'pending',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  approved_by BIGINT UNSIGNED NULL,
  approved_at TIMESTAMP NULL,
  verification_doc_path VARCHAR(1024) NULL,
  INDEX (email)
);

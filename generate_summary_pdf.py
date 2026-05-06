import os
import sys
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor

def generate_pdf():
    output_path = os.path.join(os.getcwd(), "CVIAAR_Technical_Summary.pdf")
    doc = SimpleDocTemplate(output_path, pagesize=LETTER)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=HexColor('#224abe'),
        spaceAfter=20,
        alignment=0
    )
    
    h2_style = ParagraphStyle(
        'H2Style',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=HexColor('#4e73df'),
        spaceBefore=15,
        spaceAfter=10
    )
    
    h3_style = ParagraphStyle(
        'H3Style',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=HexColor('#2d3436'),
        spaceBefore=10,
        spaceAfter=5
    )
    
    body_style = styles['BodyText']
    body_style.fontSize = 11
    body_style.leading = 14
    
    story = []
    
    # Title
    story.append(Paragraph("CVIAAR AI Biometric Attendance System", title_style))
    story.append(Paragraph("Technical Summary & Project Documentation", styles['Italic']))
    story.append(Spacer(1, 0.2 * inch))
    
    # 1. Project Overview
    story.append(Paragraph("1. Project Overview", h2_style))
    story.append(Paragraph(
        "CVIAAR is an advanced, AI-powered biometric attendance management system designed for high-accuracy identity verification and real-time attendance tracking. It leverages computer vision and machine learning to provide a seamless, secure, and automated experience for both administrators and staff members.",
        body_style
    ))
    
    # 2. Core Features
    story.append(Paragraph("2. Core Features", h2_style))
    
    story.append(Paragraph("Biometric Authentication & Kiosk", h3_style))
    features_kiosk = [
        "<b>Facial Recognition:</b> High-performance recognition using LBPH (Local Binary Patterns Histograms) with adaptive thresholding.",
        "<b>Liveness Detection:</b> Real-time blink detection (Eye Aspect Ratio - EAR) to prevent spoofing and ensure physical presence.",
        "<b>Standby Mode:</b> Intelligent camera state management that wakes up for authentication and returns to a low-resource standby state.",
        "<b>Automated Enrollment:</b> Multi-angle image capture for training new user profiles with boundary validation."
    ]
    for feat in features_kiosk:
        story.append(Paragraph(f"• {feat}", body_style))
    
    story.append(Paragraph("Analytics & Reporting", h3_style))
    features_analytics = [
        "<b>Admin Dashboard:</b> Modern, compact UI with real-time status broadcasting via Redis.",
        "<b>Interactive Visualizations:</b> Dynamic D3.js and Plotly charts showing weekly attendance trends, punctuality scores, and peak arrival windows.",
        "<b>Advanced Engine:</b> High-performance analytics engine (Rust-powered backend) for fast data processing.",
        "<b>Automated Email Reports:</b> Beautifully designed HTML performance snapshots sent to users, including motivational feedback and growth metrics.",
        "<b>Risk Prediction:</b> Algorithms to identify attendance patterns and predict potential punctuality risks."
    ]
    for feat in features_analytics:
        story.append(Paragraph(f"• {feat}", body_style))
        
    story.append(Paragraph("System Management", h3_style))
    features_mgmt = [
        "<b>Dockerized Environment:</b> Fully containerized multi-service architecture (App, Postgres, Redis).",
        "<b>Instance Lifecycle Manager:</b> PID-based locking system to prevent resource conflicts and redundant process spawning.",
        "<b>Optimized Builds:</b> Multi-stage Docker builds with significantly reduced context transfer sizes."
    ]
    for feat in features_mgmt:
        story.append(Paragraph(f"• {feat}", body_style))

    # 3. Technology Stack
    story.append(Paragraph("3. Technology Stack", h2_style))
    
    story.append(Paragraph("Languages", h3_style))
    langs = [
        "<b>Python 3.10:</b> Primary backend language for API, business logic, and computer vision integration.",
        "<b>JavaScript (ES6+):</b> Frontend logic, real-time polling, and interactive chart rendering.",
        "<b>Rust:</b> Used for the high-performance analytics core (cviaar-analytics) to ensure fast report generation.",
        "<b>HTML5 & CSS3:</b> Modern responsive layouts using Bootstrap 5 and custom CSS.",
        "<b>SQL:</b> Database querying and management for PostgreSQL."
    ]
    for lang in langs:
        story.append(Paragraph(f"• {lang}", body_style))
        
    story.append(Paragraph("Backend Frameworks & Libraries", h3_style))
    backend = [
        "<b>FastAPI:</b> High-performance web framework for the main application API.",
        "<b>SQLAlchemy:</b> ORM for database management and relationship mapping.",
        "<b>OpenCV:</b> Core library for image processing and facial recognition.",
        "<b>MediaPipe:</b> Used for high-fidelity facial landmark detection.",
        "<b>Uvicorn:</b> ASGI server for running the FastAPI application.",
        "<b>Pandas:</b> Data manipulation and analysis for the Python-side analytics engine."
    ]
    for b in backend:
        story.append(Paragraph(f"• {b}", body_style))
        
    story.append(Paragraph("Frontend Tools", h3_style))
    frontend = [
        "<b>D3.js:</b> Data-driven document manipulation for the Weekly Attendance chart.",
        "<b>Plotly.js:</b> Interactive charts for attendance velocity and status distribution.",
        "<b>Bootstrap 5:</b> Responsive UI components and layout grid.",
        "<b>Jinja2:</b> Template engine for server-side HTML rendering."
    ]
    for f in frontend:
        story.append(Paragraph(f"• {f}", body_style))
        
    story.append(Paragraph("Infrastructure & DevOps", h3_style))
    infra = [
        "<b>Docker & Docker Compose:</b> Orchestration of the application, database, and cache services.",
        "<b>PostgreSQL 15:</b> Primary relational database for user data and attendance logs.",
        "<b>Redis 7:</b> In-memory data store for real-time status broadcasting and caching."
    ]
    for i in infra:
        story.append(Paragraph(f"• {i}", body_style))

    # Build PDF
    doc.build(story)
    print(f"PDF successfully generated at: {output_path}")

if __name__ == "__main__":
    try:
        generate_pdf()
    except Exception as e:
        print(f"Error generating PDF: {e}")
        sys.exit(1)

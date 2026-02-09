from app import app

if __name__ == "__main__":
    from waitress import serve
    print("Starting Waitress WSGI server...")
    print("Local access:   http://localhost:5000")
    print("Network access: http://192.168.1.36:5000")
    serve(app, host="0.0.0.0", port=5000, threads=6)

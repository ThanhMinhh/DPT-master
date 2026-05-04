def load_sounds():
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'database.db'))
    c = conn.cursor()
    
    # 1. Truy vấn siêu dữ liệu từ CSDL
    c.execute("SELECT path, features, shape FROM sounds")
    rows = c.fetchall()
    
    sounds = []
    # 2. Vòng lặp giải mã dữ liệu
    for row in rows:
        # 2a. Khôi phục cấu trúc không gian (Shape)
        shape = eval(row[2])
        
        # 2b. Khôi phục ma trận Numpy (Deserialization)
        features = np.frombuffer(row[1], dtype=np.float64).reshape(shape)
        
        # Đóng gói thành đối tượng Sound và đưa vào mảng RAM
        sounds.append(Sound(features=features, path=row[0]))
        
    conn.close()
    return sounds

import os

def split_jsonl_file(input_filepath, output_directory, lines_per_file=30):
    """
    Tách một file .jsonl lớn thành các file nhỏ hơn.
    """
    # 1. Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_directory, exist_ok=True)
    
    # Lấy tên file gốc (không có đuôi .jsonl) để đặt tên cho các file con
    base_name = os.path.splitext(os.path.basename(input_filepath))[0]
    
    # 2. Đọc và tách file
    with open(input_filepath, 'r', encoding='utf-8') as infile:
        file_index = 1
        line_count = 0
        outfile = None
        
        for line in infile:
            # Mở một file mới cứ sau mỗi `lines_per_file` dòng
            if line_count % lines_per_file == 0:
                if outfile:
                    outfile.close()
                
                output_filename = f"{base_name}_part_{file_index}.jsonl"
                output_filepath = os.path.join(output_directory, output_filename)
                
                outfile = open(output_filepath, 'w', encoding='utf-8')
                file_index += 1
                
            # Ghi dòng hiện tại vào file con đang mở
            outfile.write(line)
            line_count += 1
            
        # Đóng file cuối cùng sau khi vòng lặp kết thúc
        if outfile:
            outfile.close()
            
    print(f"Hoàn thành! Đã tách file gốc thành {file_index - 1} file nhỏ trong thư mục '{output_directory}'.")

# --- Thực thi ---
if __name__ == "__main__":
    input_file = "blank_dataset.jsonl"
    output_dir = "blank_dataset"
    
    # Bắt đầu tách file
    split_jsonl_file(input_file, output_dir, lines_per_file=30)
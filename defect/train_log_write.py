import csv
import os
from dotenv import load_dotenv


class LogWriter:
    def __init__(self, model_path, file_name="training_log.csv"):
        # .env 파일 로드
        load_dotenv()

        # .env 파일에서 LOG_DIR 가져오기
        log_dir = model_path
        os.makedirs(log_dir, exist_ok=True)  # 로그 디렉토리 생성

        # 로그 파일 경로 설정
        log_file_path = os.path.join(log_dir, file_name)
        self.log_file = open(log_file_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.log_file)

        # CSV 파일 헤더 작성
        self.csv_writer.writerow(['num_inference_steps', 'train_loss', 'learning_rate', 'max_train_steps', 'progress'])
        self.log_file.flush()  # 헤더 작성 후 즉시 파일에 기록

    def write_log(self, num_inference_steps, train_loss, learning_rate, max_train_steps, progress):
        # 로그 데이터를 CSV에 기록
        self.csv_writer.writerow([num_inference_steps, train_loss, learning_rate, max_train_steps, progress])
        self.log_file.flush()  # 헤더 작성 후 즉시 파일에 기록

    def close(self):
        # 학습 종료 후 파일 닫기
        self.log_file.close()

import os
import subprocess
import argparse
import logging
from pathlib import Path
from typing import List


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioExtractor:
    """Класс для извлечения аудио из видеофайлов с помощью ffmpeg"""
    
    def __init__(self, output_dir: str = "extracted_audio"):
        """
        Инициализация экстрактора аудио
        
        Args:
            output_dir: Директория для сохранения извлеченного аудио
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """Проверка наличия ffmpeg в системе"""
        try:
            subprocess.run(["ffmpeg", "-version"], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE)
            logger.info("FFmpeg успешно обнаружен в системе")
        except FileNotFoundError:
            logger.error("FFmpeg не установлен! Пожалуйста, установите FFmpeg.")
            raise RuntimeError("FFmpeg не установлен. Выполните: pip install ffmpeg-python или установите через пакетный менеджер")
    
    def extract_audio(self, video_path: str, enhance_audio: bool = False) -> str:
        """
        Извлекает аудио из видеофайла и сохраняет его в формате WAV
        
        Args:
            video_path: Путь к видеофайлу
            enhance_audio: Применять ли улучшение аудио (шумоподавление)
            
        Returns:
            Путь к извлеченному аудиофайлу
        """
        video_filename = os.path.basename(video_path)
        audio_filename = os.path.splitext(video_filename)[0] + ".wav"
        audio_path = os.path.join(self.output_dir, audio_filename)
        
        # Формируем команду для ffmpeg
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",  # Отключаем видеопоток
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", "16000",  # 16kHz семплрейт
            "-ac", "1",  # Mono канал
        ]
        
        # Если нужно улучшение аудио, добавляем фильтры шумоподавления
        if enhance_audio:
            logger.info(f"Применяем шумоподавление для {video_filename}")
            cmd.extend([
                "-af", "highpass=f=200,lowpass=f=3000,afftdn=nf=-25",
            ])
        
        cmd.append(audio_path)
        
        # Запускаем ffmpeg
        logger.info(f"Извлечение аудио из {video_filename}")
        try:
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if process.returncode != 0:
                logger.error(f"Ошибка при извлечении аудио: {process.stderr}")
                raise RuntimeError(f"Ошибка FFmpeg: {process.stderr}")
                
            logger.info(f"Аудио успешно извлечено в {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"Исключение при извлечении аудио: {str(e)}")
            raise
    
    def batch_extract(self, video_dir: str, enhance_audio: bool = False) -> List[str]:
        """
        Пакетное извлечение аудио из всех видеофайлов в директории
        
        Args:
            video_dir: Директория с видеофайлами
            enhance_audio: Применять ли улучшение аудио
            
        Returns:
            Список путей к извлеченным аудиофайлам
        """
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv']
        audio_paths = []
        
        for filename in os.listdir(video_dir):
            file_path = os.path.join(video_dir, filename)
            if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in video_extensions):
                try:
                    audio_path = self.extract_audio(file_path, enhance_audio)
                    audio_paths.append(audio_path)
                except Exception as e:
                    logger.error(f"Ошибка при обработке {filename}: {str(e)}")
        
        return audio_paths


def main():
    parser = argparse.ArgumentParser(description="Извлечение аудио из видеофайлов")
    parser.add_argument("--video", type=str, help="Путь к видеофайлу")
    parser.add_argument("--video_dir", type=str, help="Директория с видеофайлами для пакетной обработки")
    parser.add_argument("--output_dir", type=str, default="extracted_audio", help="Директория для сохранения аудио")
    parser.add_argument("--enhance_audio", action="store_true", help="Применять шумоподавление")
    
    args = parser.parse_args()
    
    if not args.video and not args.video_dir:
        parser.error("Необходимо указать --video или --video_dir")
    
    extractor = AudioExtractor(args.output_dir)
    
    if args.video:
        if not os.path.isfile(args.video):
            raise FileNotFoundError(f"Видеофайл не найден: {args.video}")
        audio_path = extractor.extract_audio(args.video, args.enhance_audio)
        print(f"Аудио извлечено в: {audio_path}")
    
    if args.video_dir:
        if not os.path.isdir(args.video_dir):
            raise NotADirectoryError(f"Директория не найдена: {args.video_dir}")
        
        audio_paths = extractor.batch_extract(args.video_dir, args.enhance_audio)
        print(f"Извлечено {len(audio_paths)} аудиофайлов")
        for path in audio_paths:
            print(f"  - {path}")


if __name__ == "__main__":
    main()

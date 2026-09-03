# scripts/split_large_parquet.py
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path


# Разбивает огромный Parquet файл на части БЕЗ загрузки всего в RAM. Читает по кусочкам (batch_size) и сохраняет в part_*.parquet.
def split_large_parquet(
    input_path: Path,
    rows_per_part: int = 50_000,  # ← МАЛЕНЬКИЙ РАЗМЕР!
    batch_size: int = 10_000,      # ← ЧИТАЕМ МАЛЕНЬКИМИ КУСОЧКАМИ!
):

    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"Файл не найден: {input_path}")
        return
    
    file_size_gb = input_path.stat().st_size / 1024**3
    print(f"Разбиваем {input_path.name} ({file_size_gb:.2f} ГБ)")
    print(f"Размер части: {rows_per_part:,} строк")
    print(f"Размер батча: {batch_size:,} строк (для экономии RAM)")
    
    try:
        # ОТКРЫВАЕМ ФАЙЛ (НЕ ЗАГРУЖАЕМ В RAM!)
        parquet_file = pq.ParquetFile(input_path)
        total_rows = parquet_file.metadata.num_rows
        num_parts = (total_rows + rows_per_part - 1) // rows_per_part
        
        print(f"   Всего строк: {total_rows:,}, частей: {num_parts}")
        print("   Чтение по батчам...")
        
        parent_dir = input_path.parent
        current_part = []
        part_index = 0
        rows_in_current_part = 0
        
        # ЧИТАЕМ ПО МАЛЕНЬКИМ БАТЧАМ
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            # Преобразуем батч в таблицу
            table = pa.Table.from_batches([batch])
            
            # Разбиваем батч на строки
            for i in range(table.num_rows):
                row = table.slice(i, 1)
                current_part.append(row)
                rows_in_current_part += 1
                
                # Если набрали нужное количество строк — сохраняем
                if rows_in_current_part >= rows_per_part:
                    # Объединяем строки в таблицу
                    combined_table = pa.concat_tables(current_part)
                    output_path = parent_dir / f"part_{part_index:04d}.parquet"
                    pq.write_table(combined_table, output_path)
                    
                    # Очищаем
                    current_part = []
                    rows_in_current_part = 0
                    part_index += 1
                    
                    del combined_table
                    
                    if part_index % 10 == 0:
                        print(f"   Сохранено {part_index}/{num_parts} частей")
            
            # ОЧИЩАЕМ ПАМЯТЬ ПОСЛЕ КАЖДОГО БАТЧА
            del table
            
        # Сохраняем остаток
        if current_part:
            combined_table = pa.concat_tables(current_part)
            output_path = parent_dir / f"part_{part_index:04d}.parquet"
            pq.write_table(combined_table, output_path)
            part_index += 1
            del combined_table
        
        # ЗАКРЫВАЕМ ФАЙЛ
        del parquet_file
        
        print(f"Сохранено {part_index} частей")
        
        # Удаляем оригинал
        try:
            input_path.unlink()
            print(f"Удалён оригинал {input_path.name}")
        except Exception as e:
            print(f"Не удалось удалить оригинал: {e}")
            
    except Exception as e:
        print(f"Ошибка: {e}")


# Разбивает все большие Parquet файлы в папке
def split_all_large_files(data_dir: Path, min_size_gb: float = 0.5):
    data_dir = Path(data_dir)
    
    # Ищем все Parquet файлы
    parquet_files = list(data_dir.rglob("*.parquet"))
    
    print(f"Найдено {len(parquet_files)} Parquet файлов\n")
    
    for parquet_file in parquet_files:
        size_gb = parquet_file.stat().st_size / 1024**3
        
        if size_gb < min_size_gb:
            print(f"Пропускаем {parquet_file.relative_to(data_dir)} ({size_gb:.2f} ГБ) — маленький")
            continue
        
        # РАЗБИВАЕМ
        split_large_parquet(
            input_path=parquet_file,
            rows_per_part=50_000,   # 50k строк на часть (~100-200 МБ)
            batch_size=10_000       # читаем по 10k строк
        )
        
        print()  # пустая строка


if __name__ == "__main__":
    DATA_DIR = Path("E:/AI/Projects/my_llm/data")
    
    answer = input("Разбить все файлы больше 0.5 ГБ? (y/n): ")
    if answer.lower() == "y":
        split_all_large_files(
            data_dir=DATA_DIR,
            min_size_gb=0.5
        )
        print("\nГотово!")
    else:
        print("Отмена")
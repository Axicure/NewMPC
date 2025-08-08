import csv
import random

def generate_csv_files():
    total_rows = 4050
    ranges = [5, 10, 15, 20]

    for r in ranges:
        filename = f'container_machine_id_{r}.csv'
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(1, total_rows + 1):
                random_value = random.randint(1, r)
                writer.writerow([i, random_value])
        print(f'Generated: {filename}')

if __name__ == '__main__':
    generate_csv_files()

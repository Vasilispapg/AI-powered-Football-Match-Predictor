import os
import csv


folder=os.listdir('matches_detailed_processed')
data=[]
for file_name in folder:
    if file_name.endswith(".csv"):
        with open('matches_detailed_processed/'+file_name, 'r',encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
                    
with open('merged/merged_output_detailed.csv', 'w',encoding='utf-8',newline='') as f:
    writer = csv.writer(f)
    for item in data:
        writer.writerow(item)
                        
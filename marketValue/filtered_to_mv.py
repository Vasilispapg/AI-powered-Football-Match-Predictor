
import csv


def f_to_mv(input_filtered,output):
     
    teams={}
    with open(input_filtered, "r", newline="", encoding="utf-8") as output_filtered:
        reader = csv.reader(output_filtered)
        next(output_filtered)
        for row in reader:
            teams[row[2]]='0'
            teams[row[4]]='0'
            
    with open(output, "r", newline="", encoding="utf-8") as output_filtered:
        reader=csv.reader(output_filtered)
        next(output_filtered)
        for item in reader:
            teams[item[0]]=item[1]
    
    with open(output,'w',newline='',encoding='utf-8') as output_file:
        writer=csv.writer(output_file)
        writer.writerow(['Team Name','Market Value'])
        for key,value in teams.items():
            writer.writerow([key,value])
    print("END f_to_mv")
            
if __name__ == "__main__":
    f_to_mv('filter/output_filtered.csv','marketValue/teams_market_value.csv')
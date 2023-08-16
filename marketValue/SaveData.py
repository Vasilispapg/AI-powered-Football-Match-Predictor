import json
import csv


def add_json_to_csv():
    # Read JSON data
    lista=[]
    with open('marketValue/temp.json', 'r',encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        for id, data in json_data.items():
            team_name=list(data)
            lista.append([team_name[0],data[team_name[0]]])

    # # Write JSON data into a CSV file
    with open('marketValue/teams_market_value.csv', 'w', newline='',encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Team Name", "Market Value"])
        writer.writerows(lista)
        
                
    # Open the file in write mode to clear its contents
    with open('marketValue/temp.json', "w") as json_file:
        # Write an empty JSON object to the file
        json_file.write("{}")
         
if __name__ == "__main__":
    add_json_to_csv()

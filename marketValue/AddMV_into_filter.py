import csv
import pandas as pd



def addDataIntoMV():
    ''' The `addDataIntoMV` function is reading data from two CSV files, `teams_market_value_fix.csv`
    and `output_filtered.csv`. It then matches the team names from the second file with the team
    names from the first file and appends the corresponding market value to each line in the second
    file. Finally, it writes the updated data back to the `output_filtered.csv` file.
    '''
    data=[]
    data_filtered=[]
    with open('marketValue/teams_market_value_fix.csv', 'r',encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for line in csv_reader:
            data.append(line)
            
    with open('filter/output_filtered.csv', 'r',encoding='utf-8') as filteredFile:
        csv_reader = csv.reader(filteredFile)
        next(csv_reader)
        for line in csv_reader:
            data_filtered.append(line[:-2])

    for team,value in data:
        for team_filtered in data_filtered:
                if(team_filtered[2]==team):
                    team_filtered.append(value)
                
    for team,value in data:
        for team_filtered in data_filtered:
            if(len(team_filtered)>4):
                if(team_filtered[4]==team):
                    team_filtered.append(value)

    with open('filter/output_filtered.csv', 'w',encoding='utf-8',newline='') as filteredFile:
        csv_writer = csv.writer(filteredFile)
        csv_writer.writerow(['Competition','Country',"Home Team","Home Score","Away Team","Away Score","Date","MV Home Team","MV Away Team"])
        for line in data_filtered:
            csv_writer.writerow(line)


addDataIntoMV()
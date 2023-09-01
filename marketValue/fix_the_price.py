import csv
import pandas as pd

def expand_number_in_k(value):
    suffixes = {
        'k': 1e-3,
        'm': 1,
        'b': 1e3,
    }
    
    value = value.lower().strip()
    for suffix, multiplier in suffixes.items():
        if value.endswith(suffix):
            numeric_part = value[:-1]
            try:
                numeric_value = float(numeric_part.replace(',', ''))
                expanded_value = numeric_value * multiplier
                return expanded_value
            except ValueError:
                return None  # Invalid input
    try:
        return float(value.replace(',', ''))
    except ValueError:
        return None  # Invalid input
    
#if you find 0 make it median 2.28M
def fix_the_price():
    """
    The function "fix_the_price" reads a CSV file containing team names and market values, converts the
    market values to numeric format, calculates the median value, and replaces any zero values with the
    calculated median value.
    """
    data=[]
    with open('marketValue/teams_market_value.csv','r',encoding='utf-8') as f:
        reader=csv.reader(f)
        next(reader)
        for row in reader:
            num=expand_number_in_k(row[1])
            if(num!=0 and num):
                data.append([row[0],round(num,3)])
            else:
                data.append([row[0],0])
                
        # Create a DataFrame from the collected data
        columns = ["Team Name", "Market Value"]
        df = pd.DataFrame(data, columns=columns)

        # Convert "Market Value (M)" column to numeric
        df["Market Value"] = pd.to_numeric(df["Market Value"])
         # Calculate mid, median, and mean values
        mid_value = round(df['Market Value'].median()-int(str(df['Market Value'].median()).split('.')[0]),3)
        
        #keep only the .xxx part of the median value and replace the 0 values with it
        # because the median value is 2.28M
        # and the teams without values are small teams
        for idx, (team, value) in enumerate(data):
            if value == 0:
                data[idx][1] = mid_value
                
            
    with open('marketValue/teams_market_value.csv', "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Team Name", "Market Value"])
        writer.writerows(data)
        
if __name__ == "__main__":
    fix_the_price()

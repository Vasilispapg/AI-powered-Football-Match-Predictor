import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from fix_the_price import fix_the_price
import json
from SaveData import add_json_to_csv
from filtered_to_mv import f_to_mv
from AddMV_into_filter import addDataIntoMV
from selenium.common.exceptions import TimeoutException,NoSuchElementException


def find_string_in_list(search_string, string_list):
    for item in string_list:
        if search_string.lower() in item.text.strip().lower():
            return item
    return None  # Return None if the search string is not found in any item

def team_exists_in_csv(team_name, filename):
    """
    The function checks if a team name exists in a CSV file.
    
    :param team_name: The name of the team you want to check if it exists in the CSV file
    :param filename: The filename parameter is the name of the CSV file that you want to check for the
    existence of a team
    :return: a boolean value. It returns True if the team name exists in the CSV file specified by the
    filename parameter, and False otherwise.
    """
    with open(filename, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if row and row[0] == team_name:
                return True
    return False

def save_market_values_to_csv(values_dict, filename):
    """
    The function saves a dictionary of team names and market values to a CSV file.
    
    :param values_dict: values_dict is a dictionary where the keys are team names and the values are
    their corresponding market values
    :param filename: The filename parameter is a string that represents the name of the CSV file where
    the market values will be saved
    """
    with open(filename, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        for team_name, market_value in values_dict.items():
            writer.writerow([team_name, market_value])

def importJSON(filename,data):
    with open(filename, 'w',encoding='utf-8') as f:
        json.dump(data, f)
        
def loadJSON(filename):
    with open(filename, 'r',encoding='utf-8') as f:
        return json.load(f)
              
def fetchMarketValues(team_list,inputFile=None):
    """
    The function `fetchMarketValues` uses Selenium and Firefox to scrape market values of football teams
    from Transfermarkt website and stores them in a dictionary.
    
    :param team_list: The `team_list` parameter is a list of dictionaries. Each dictionary represents a
    team and its market value. The dictionary has two keys: "Team Name" and "Market Value". The "Team
    Name" key holds the name of the team, and the "Market Value" key holds the market
    :param inputFile: The `inputFile` parameter is the path to the CSV file where the team names and
    market values will be written
    """
    
    temp_file=loadJSON('marketValue/temp.json')

    # Set up Selenium options for Firefox (GeckoDriver)
    options = Options()
    options.headless = True  # Run Firefox in headless mode
    options.set_preference("network.cookie.cookieBehavior", 2)  # Disable cookies

    # Create a Firefox driver instance using GeckoDriver
    driver = webdriver.Firefox(options=options, executable_path='webScrapper/geckodriver.exe')

    # Base URL for Transfermarkt
    base_url = "https://www.transfermarkt.com"

    # Dictionary to store market values
    market_values = []
    id=0
    for team_name,value in team_list:
        id+=1
        if(id%10==0):
            importJSON('marketValue/temp.json',temp_file)
            
        if(value!='0'):            
            market_values.append([team_name,value])
            temp_file[str(id)]={team_name:value}
            continue
        
        team_name_search=team_name
        if(' ' in team_name):
            team_name_search = team_name.replace(' ', '%20')
        search_url = f"{base_url}/schnellsuche/ergebnis/schnellsuche?query={team_name_search}"

        try:
            # Open the search URL in the browser
            driver.get(str(search_url))

            # Wait for search results to load
            try:
                main_element = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, "main")))
                try:
                    h2_elements = main_element.find_elements(By.CSS_SELECTOR, "h2.content-box-headline")
                except NoSuchElementException:
                    pass
                
                club_element = find_string_in_list('club', h2_elements)
               
                if club_element!=None:
                    print(f"Waiting for search results for {team_name} id={id}...")
                    try:
                        # Find the parent element of the h2 element
                        parent_element = club_element.find_element(By.XPATH, "..")
                        # Find the rechts element within the parent element
                        rechts_element = parent_element.find_element(By.CLASS_NAME, "rechts")
                        market_value_text = rechts_element.text.strip()
                        if '€' in market_value_text:
                            print(f"Found market value for {team_name}: {market_value_text}")
                            market_values.append([team_name,market_value_text.replace('€', '')])
                            temp_file[str(id)]={team_name:market_value_text.replace('€', '')}
                        else:
                            market_values.append([team_name,'0'])
                            temp_file[str(id)]={team_name:'0'}

                    except Exception as e:
                        print(f"Error while extracting market value for {team_name}: {e}")
                        market_values.append([team_name,'0'])
                        temp_file[str(id)]={team_name:'0'}


                else:
                    market_values.append([team_name,'0'])
                    temp_file[str(id)]={team_name:'0'}

                # print(market_values[team_name])
            except TimeoutException:
                print("Timeout waiting for search results.")
                
        except Exception as e:
            if "403" in str(e):
                print(f"403 Forbidden Error: {team_name}")
            else:
                print(f"Error for {team_name}: {e}")
                
            # Write the updated data back to the CSV input file
                
        # Write the updated data back to the CSV file
    with open(inputFile, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Team Name", "Market Value"])
        writer.writerows(market_values)
            
        # 0=2074 
        # new 0 = 

    # Close the browser
    driver.quit()
    
def checkagain(input_file):

    # Read the CSV file and populate the data array
    data_array = []    
    with open(input_file, "r", newline="", encoding="utf-8") as input_csv:
        reader = csv.reader(input_csv)
        for row in reader:
            for row in reader:
                data_array.append(row)
        fetchMarketValues(data_array,input_file)
    print('END checkagain')
        
if __name__ == "__main__":
    f_to_mv('filter/output_filtered.csv','marketValue/teams_market_value.csv')
    checkagain('marketValue/teams_market_value.csv')
    add_json_to_csv()
    fix_the_price()
    addDataIntoMV()
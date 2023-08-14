import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time


def team_exists_in_csv(team_name, filename):
    with open(filename, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if row and row[0] == team_name:
                return True
    return False

def save_market_values_to_csv(values_dict, filename):
    with open(filename, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        for team_name, market_value in values_dict.items():
            writer.writerow([team_name, market_value])

            

# Read the input CSV to get unique team names
team_names = set()
with open("filter/output_filtered.csv", "r", newline="", encoding="utf-8") as input_csv:
    reader = csv.DictReader(input_csv)
    for row in reader:
        team_names.add(row["Home Team"])
        team_names.add(row["Away Team"])
        
team_name_chunks = [list(team_names)[i:i + 10] for i in range(0, len(team_names), 10)]


# Set up Selenium options for Firefox (GeckoDriver)
options = Options()
options.headless = True  # Run Firefox in headless mode
options.set_preference("network.cookie.cookieBehavior", 2)  # Disable cookies

# Create a Firefox driver instance using GeckoDriver
driver = webdriver.Firefox(options=options, executable_path='webScrapper/geckodriver.exe')

# Base URL for Transfermarkt
base_url = "https://www.transfermarkt.com"

# Dictionary to store market values
market_values = {}
print(len(team_names))

for chunk in team_name_chunks:
    for team_name in chunk:
        # Construct the search URL
        if(team_exists_in_csv(team_name, "marketValue/teams_market_value.csv")):
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
                club_element = driver.find_element(By.CSS_SELECTOR, "h2.content-box-headline")
                if "club" in club_element.text.lower():
                    print(f"Waiting for search results for {team_name}...")
                    WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CLASS_NAME, "rechts")))
                    
                    # Find the parent element of the h4 element
                    parent_element = club_element.find_element(By.XPATH, "..")
                    
                    # Find the rechts element within the parent element
                    rechts_element = parent_element.find_element(By.CLASS_NAME, "rechts")
                    
                    if '€' in rechts_element.text.strip():
                        market_values[team_name] = rechts_element.text.strip().replace('€', '')
                    else:
                        market_values[team_name] = '0'
                # print(market_values[team_name])
            except TimeoutException:
                print("Timeout waiting for search results.")
                
        except Exception as e:
            if "403" in str(e):
                print(f"403 Forbidden Error: {team_name}")
            else:
                print(f"Error for {team_name}: {e}")
        
        # Save market values for this chunk to a CSV file
        save_market_values_to_csv(market_values, f"marketValue/teams_market_value.csv")
        market_values.clear()  # Clear dictionary for the next chunk

# Close the browser
driver.quit()

# Print the market values
for team_name, market_value in market_values.items():
    print(f"{team_name}: {market_value}")
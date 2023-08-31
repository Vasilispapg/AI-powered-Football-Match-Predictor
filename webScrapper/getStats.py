
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
import csv
import os
import json
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures 



# this is a way to run the script from a batch file and from python script
# to run again when the program stops for some reason
if os.environ.get("RUNNING_FROM_BATCH"):
    # Running from batch, set batch-specific paths
    geckodriver_path = r'C:\Users\vasil\Downloads\football\webScrapper\geckodriver.exe'
    temp_json_path = r'C:\Users\vasil\Downloads\football\webScrapper\temp'
    folder_path = r'C:\Users\vasil\Downloads\football\matches_detailed'
    output_folder = r'C:\Users\vasil\Downloads\football\matches_detailed_processed'
else:
    # Running as Python script, set regular paths
    geckodriver_path = "webScrapper/geckodriver.exe"
    temp_json_path = "webScrapper/temp"
    folder_path = "matches_detailed"
    output_folder = "matches_detailed_processed"
    
    
def getStats(driver):
    """
    The `getStats` function takes a Selenium WebDriver object as input, navigates to a match details
    page, and extracts the statistics from the page into a dictionary.
    
    :param driver: The "driver" parameter is an instance of a Selenium WebDriver. It is used to interact
    with the web page and retrieve the necessary HTML data
    :return: The function `getStats` returns a dictionary containing the statistics extracted from the
    HTML data. The keys of the dictionary are the stat categories (extracted from the h3 elements), and
    the values are lists of stat values (extracted from the li elements within each stat category).
    """
    # Assuming you have already navigated to the match details page and have the necessary HTML data available

    # Find all the h3 elements containing stat categories
    
    stat_categories = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.TAG_NAME, "h3"))
    )
    # Initialize a dictionary to store the stats
    stats_dict = {}

    # Iterate through each stat category

    for i,category in enumerate(stat_categories):
        if(i<2):
            continue
        else:
            ul_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, f"/html/body/div/div/div[2]/main/div[2]/div[{i}]/ul")))        
        
            # Find all li elements within the ul element
            li_elements = ul_element.find_elements(By.TAG_NAME, "li")
            
            # Iterate through each li element and extract stat name and value
            for li_element in li_elements:
                stat_name = None
                stat_values = []
                try:
                    stat_name = li_element.find_element(By.TAG_NAME, "h4").text
                    
                    stat_value = li_element.find_elements(By.CLASS_NAME, "stats-group-row_data__iIVZq")
                    for value in stat_value:
                        stat_values.append(value.text)
                        
                    stats_dict[str(stat_name)] = stat_values

                except Exception as e:
                    print(e)

        return stats_dict
     
def getVotes(driver,home,away):
    """
    The function `getTheVotes` takes a web driver and a list of team names as input, navigates to a
    match details page, extracts the votes for each team from the HTML data, and returns a list
    containing the team names and their corresponding votes.
    
    :param driver: The "driver" parameter is an instance of a web driver, such as Selenium's WebDriver,
    that is used to interact with a web browser and navigate to web pages
    :param team_names: A list of team names for which you want to get the votes
    :return: a 2D array containing the team names and their corresponding votes.
    """
        # Assuming you have already navigated to the match details page and have the necessary HTML data available

    # Find the predictor container
    predictor_container = driver.find_element(By.CLASS_NAME, "predictor_predictor__dAM6K")

    # Find the list of result wrappers
    result_wrappers = predictor_container.find_elements(By.CLASS_NAME, "results_result-wrapper__O6nyX")

    # Initialize lists to store team names and votes
    team_votes = []

    # Iterate through each result wrapper
    for result_wrapper in result_wrappers:
        
        try:
            votes_element = result_wrapper.find_element(By.CLASS_NAME, "results_votes__aFpgS")
            team_votes_text = votes_element.text.split(' ')[0]
            team_votes.append(team_votes_text)
        except:
            continue

    # Create the final array
    result_array = [[home,'Draw',away], team_votes]

    # Print the final array
    return result_array

def importJSON(filename,data):
    with open(filename, 'w',encoding='utf-8') as f:
        json.dump(data, f)
        
def loadJSON(filename):
    if(os.path.exists(filename)==False):
        return {}
    with open(filename, 'r',encoding='utf-8') as f:
        return json.load(f)
  
def proccess(driver,file_path):
 
    data=[]
    counter=1
    # Process each URL and store additional information
    output_csv_filename = file_path.replace("matches_detailed","matches_detailed_processed").split('\\')[-1].split('.')[0]
    output_csv_filepath = file_path.replace("matches_detailed","matches_detailed_processed")
    temp_file=loadJSON(temp_json_path+file_path.replace("matches_detailed","matches_detailed_processed").split('\\')[-1].split('.')[0]+'.json')
    
    for key,item in temp_file.items():
        data.append(item)

    if(os.path.exists(output_csv_filepath)):
        return

    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            print(row['Home Team'], row['Away Team'])
            if(str(counter) in temp_file):
                counter+=1
                continue
            
            url = row['URL']  # Assuming you have a 'URL' column in your CSV
            driver.get(url)
            
            row['Stats']={}
            row['Votes']=[]

            try:
                row['Stats']=getStats(driver)
                row['Votes']=getVotes(driver, row['Home Team'], row['Away Team'])
            except:
                row['Stats']={}
                row['Votes']=[]
                        
            data.append(row)
            temp_file[counter]=row
            counter+=1
            if(counter%10==0):
                importJSON(temp_json_path+output_csv_filename+'.json',temp_file)

    # Write the data to a new CSV file
    with open(output_csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ['Competition', 'Country', 'Home Team', 'Home Score', 'Away Team', 'Away Score', 'Date', 'URL', 'Votes','Stats']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(data)
        
    os.remove(temp_json_path+output_csv_filename+'.json')
    # clear the temp file

    print(f"Data has been processed and saved to {output_csv_filename}")

def main(provided_file_list):
    # Initialize a Firefox driver instance
    options = webdriver.FirefoxOptions()
    options.headless = True
    profile=FirefoxProfile()
    options.set_preference("network.cookie.cookieBehavior", 0)
    driver = webdriver.Firefox(options=options, executable_path=geckodriver_path,firefox_profile=profile)
    for file_name in provided_file_list:
        if file_name.endswith(".csv"):
            print(f"Processing data from: {file_name}")
            file_path = os.path.join(folder_path, file_name)
            processed_file_path = proccess(driver,file_path)
            print(f"Processed data saved to: {processed_file_path}")
    driver.quit()



# Use ThreadPoolExecutor for parallel processing

os.makedirs(output_folder, exist_ok=True)
file_list = os.listdir(folder_path)
file_list=sorted(file_list)

with ThreadPoolExecutor(max_workers=3) as executor:
    futures1=executor.submit(main, file_list[int((len(file_list)/2)):])
    futures2=executor.submit(main, file_list[:int((len(file_list)/2))])
    futures3=executor.submit(main, file_list[int((len(file_list)/4)):])
    
    futures1.result()
    futures2.result()
    futures3.result()


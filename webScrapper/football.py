from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import TimeoutException,NoSuchElementException,StaleElementReferenceException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor
import time
import logging


def removeCookies(driver):
    try:
        wait = WebDriverWait(driver, 5)
        wait.until(EC.presence_of_element_located((By.XPATH, "/html/body/div/div[2]/div[3]/div[2]/button")))
        yes_button = driver.find_element(By.XPATH, "/html/body/div/div[2]/div[3]/div[2]/button")
        yes_button.click()
        logging.info("Clicked the 'Remove Cookies' button.")
    except NoSuchElementException:
        # The button doesn't exist, so you can skip this step
        logging.warning("The Cookies button was not found. Skipping...")
    except Exception as e:
        # Log the error if there is any other issue
        logging.error(f"An error occurred: {str(e)}")

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

        ul_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, f"#__next > div > div.layout_layout__aM4CA > main > div.stats_stats__JdhGQ > div:nth-child({i+2}) > ul")))        
        
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

def getTheVotes(driver,team_names):
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
    result_array = [team_names, team_votes]

    # Print the final array
    return result_array

def fetch(urls):
    """
    The function `fetch` uses Selenium to scrape data from multiple URLs, extracts match data, and saves
    it to CSV files.
    
    :param urls: The `urls` parameter is a list of URLs that you want to fetch data from. Each URL
    represents a webpage from which you want to extract data
    """
    # Configure Selenium to use Chrome browser
    # Set up Selenium options for Firefox (GeckoDriver)
    options = Options()
    options.headless = False  # Run Firefox in headless mode
    # options.set_preference("network.cookie.cookieBehavior", 2)  # Disable cookies

    # Create a Firefox driver instance using GeckoDriver
    driver = webdriver.Firefox(options=options, executable_path='webScrapper/geckodriver.exe')

    for url in urls:
        print(url)

        print(f"Fetching data for URL: {url}")
        
        retry_count = 0
        max_retries = 3
        
        if os.path.exists(f"matches_data_{url.split('/')[-1]}.csv"):
            print(f"CSV file for URL {url} already exists. Skipping...")
            continue

        while retry_count < max_retries:
            try:
                # Initialize a list to store match data
                matches_data = []
                driver.get(url)
                removeCookies(driver)
                wait = WebDriverWait(driver, 45)
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "row_row__UQmGm")))
                # Find all match rows
                competitions=driver.find_elements(By.CLASS_NAME, "competition_competition__s2ULZ")
                
                # Iterate through each match row
                date=url.split("/")[-1]
                for comp in competitions:
                    match_rows = comp.find_elements(By.CLASS_NAME, "row_row__UQmGm")
                    for match_row in match_rows:
                        try:
                            competition_element = comp.find_element(By.CSS_SELECTOR, "a.competition_name__YEMb_")
                            competition = competition_element.text.strip().split(' - ')[1].strip()
                            country = competition_element.text.strip().split(' - ')[0].strip()
                        except:
                            competition = "N/A"
                            country = "N/A"
                        try:
                            home_team_element = match_row.find_element(By.CSS_SELECTOR, "div.team_team-a__2YS_9 h4.name_name__YzgHa")
                            home_team = home_team_element.text.strip()
                        except:
                            home_team = "N/A"

                        try:
                            away_team_element = match_row.find_element(By.CSS_SELECTOR, "div.team_team-b__YaeU1 h4.name_name__YzgHa")
                            away_team = away_team_element.text.strip()
                        except:
                            away_team = "N/A"

                        try:
                            match_result_element = match_row.find_element(By.CLASS_NAME, "result_score__Dh4zx")
                            match_result = match_result_element.text.split("-")
                            home_score = match_result[0].strip()
                            away_score = match_result[1].strip()
                        except:
                            home_score = "N/A"
                            away_score = "N/A"
                        try:
                            match_link_element = match_row.find_element(By.CLASS_NAME, "row_match-link__cmwZt")
                            link=match_link_element.get_attribute('href')
                            # print(match_link_element.get_attribute('href'))   
                            # match_link_element.click()
                            
                            # WebDriverWait(driver, 10)  # Adjust the timeout as needed
                            # stats={}
                            # votes=[]
                            # try:
                            #     votes=getTheVotes(driver,[home_team,'Draw',away_team])
                            # except Exception as e:
                            #     print('vote:',e)
                                
                            # try:
                            #     stats=getStats(driver)
                            # except Exception as e:
                            #     print('stats:',e)
                                
                            # driver.back()
                            # wait = WebDriverWait(driver, 45)
                            # competitions=driver.find_elements(By.CLASS_NAME, "competition_competition__s2ULZ")
                            # match_rows = comp.find_elements(By.CLASS_NAME, "row_row__UQmGm")
                            
                        except Exception as e:
                            print(e)
                            match_link_element=None

                            
                        # print(competition, country, home_team, home_score, away_team, away_score, date,votes,stats)
                        matches_data.append([competition, country, home_team, home_score, away_team, away_score, date,link])
                
                # If successful, break out of the retry loop
                break
            except TimeoutException:
                print(f"Timeout occurred. Retrying... ({retry_count + 1}/{max_retries})")
                retry_count += 1
                if retry_count == max_retries:
                    print("Maximum number of retries reached. Skipping this URL.")
                    break
                else:
                    driver.refresh()

       

        # Write the data into a CSV file for this URL
        csv_filename = f"matches_detailed/matches_data_{date}.csv"
        with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Competition", "Country", "Home Team", "Home Score", "Away Team", "Away Score", "Date"])
            csv_writer.writerows(matches_data)

        print(f"Data for URL {url} has been extracted and saved to {csv_filename}")

    # Close the browser
    driver.quit()
        
def generate_urls(start_date, end_date):
    urls = []
    current_date = start_date

    while current_date <= end_date:
        formatted_date = current_date.strftime("%Y-%m-%d")
        url = f"https://www.goal.com/en/results/{formatted_date}"
        urls.append(url)
        current_date += timedelta(days=1)

    return urls

# Define the date range
start_date = datetime(2022, 10, 9)
end_date = datetime(2023, 8, 27)

# Generate URLs for normal date order
urls_to_fetch_normal = generate_urls(start_date, end_date)

# Generate URLs for reverse date order and reverse the list
urls_to_fetch_reverse = generate_urls(start_date, end_date)
urls_to_fetch_reverse.reverse()


# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor(max_workers=4) as executor:
    future_normal = executor.submit(fetch, urls_to_fetch_normal)
    future_reverse = executor.submit(fetch, urls_to_fetch_reverse)
    
    # Wait for the tasks to complete
    future_normal.result()
    future_reverse.result()
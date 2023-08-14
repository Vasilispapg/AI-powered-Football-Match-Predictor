from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import csv
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor


def fetch(urls):
    # Configure Selenium to use Chrome browser
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run Chrome in headless mode
    # options.add_argument("--window-size=1920x1080")
    # Replace this path with the path to your chromedriver executable
    chromedriver_path = "geckodriver.exe"

    # Initialize the Chrome driver
    driver = webdriver.Chrome(executable_path=chromedriver_path, options=options)

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
                driver.get(url)
                wait = WebDriverWait(driver, 15)
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "row_row__UQmGm")))
                # Rest of your data extraction code...
                
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

        # Find all match rows
        competitions=driver.find_elements(By.CLASS_NAME, "competition_competition__s2ULZ")
        

        # Initialize a list to store match data
        matches_data = []

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

                matches_data.append([competition, country, home_team, home_score, away_team, away_score, date])

        # Write the data into a CSV file for this URL
        csv_filename = f"matches_data_{date}.csv"
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
end_date = datetime(2023, 8, 12)

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
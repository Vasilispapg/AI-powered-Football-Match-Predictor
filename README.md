# AI Powered Football Match Predictor README

![Football Data Analysis](https://images.pexels.com/photos/46798/the-ball-stadion-football-the-pitch-46798.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)

Welcome to the Football Data Analysis repository! This README provides an extensive overview of a powerful set of Python scripts for football data analysis. These versatile scripts allow you to collect, filter, and analyze football match data, including statistics, market values, and performance metrics. Whether you're a data enthusiast, a sports analyst, or a football fan, this repository provides valuable tools to gain insights from football match data.

## Table of Contents

- [Scripts Overview](#scripts-overview)
- [Usage Instructions](#usage-instructions)
- [Project Structure](#project-structure)
- [Scripts Details](#scripts-details)
  - [Data Collection](#data-collection)
    - [`getStats.py`](#getstatspy)
    - [`football.py`](#footballpy)
  - [Data Filtering](#data-filtering)
    - [`filterData.py`](#filter-datapy)
  - [Performance Analysis](#performance-analysis)
    - [`performance.py`](#performancepy)
  - [Market Value Extraction](#market-value-extraction)
    - [`saveData.py`](#savedatapy)
    - [`marketValue.py`](#marketvaluepy)
  - [Data Manipulation](#data-manipulation)
    - [`fix_the_price.py`](#fix-the-pricepy)
    - [`filter_to_mv.py`](#filter-to-mvpy)
    - [`addmv_to_filter.py`](#addmv-to-filterpy)
  - [Utility](#utility)
    - [`merge.py`](#mergepy)
  - [Machine Learning Model](#machine-learning-model)
    - [`train.py`](#train-modelpy)
    - [`predict.py`](#predict-winnerpy)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

The project is organized into the following folders:

- **merged**: Contains merged CSV files.
- **marketValue**: Includes scripts and data related to market value extraction.
- **filter**: Contains scripts and data for filtering match data.
- **performance**: Includes scripts for performance analysis.
- **matches_detailed**: Contains raw match data.
- **matches_detailed_processed**: Contains processed match data with statistics and votes by users.
- **model**: Reserved for the machine learning model.
- **webScrapper**: Contains web scraping-related files.

## Scripts Overview

### Data Collection

1. #### `getStats.py`
   - **Description**: Collects detailed football match statistics from a website using Selenium.
   - **Output**: Saves data as JSON and CSV files.
   - **Usage**: Provides raw statistical data for further analysis.

2. #### `football.py`
   - **Description**: Collects football match data, calculates win probabilities, and generates a performance dataset.
   - **Output**: Produces a dataset with win probabilities and other match-related details.
   - **Usage**: Analyze teams' performance and predict match outcomes.

### Data Filtering

3. #### `filterData.py`
   - **Description**: Filters football match data, categorizing competitions as cup, friendly, or league. Removes rows with incomplete data.
   - **Output**: A clean and structured dataset suitable for analysis.
   - **Usage**: Prepare data for performance analysis.

### Performance Analysis

4. #### `performance.py`
   - **Description**: Calculates win probabilities based on market values and teams' recent performance.
   - **Output**: Adds win probability metrics to the dataset.
   - **Usage**: Analyze teams' performance and predict match outcomes more accurately.

### Market Value Extraction

5. #### `saveData.py`
   - **Description**: Reads JSON data and converts it into a CSV file.
   - **Output**: Converts JSON data into a CSV format for compatibility.
   - **Usage**: Prepare data for market value extraction.

6. #### `marketValue.py`
   - **Description**: Extracts market values of football teams using web scraping.
   - **Output**: Generates a CSV file with market values.
   - **Usage**: Collect market value data for football teams.

### Data Manipulation

7. #### `fix_the_price.py`
   - **Description**: Fixes market value data by converting values to a uniform format and handling zero values.
   - **Output**: Updated market value data with uniform formatting.
   - **Usage**: Ensure consistency in market value data.

8. #### `filter_to_mv.py`
   - **Description**: Transfers market value data to the filtered dataset, linking teams with their market values.
   - **Output**: Merges filtered match data with market values.
   - **Usage**: Combine match data with market values for comprehensive analysis.

9. #### `addmv_to_filter.py`
   - **Description**: Appends market values to the filtered dataset using team name matching.
   - **Output**: Updated filtered dataset with appended market values.
   - **Usage**: Enhance match data with market value information.

## Machine Learning Model

### Training the Model

10. #### `train_model.py`
    - **Description**: Builds and trains a neural network model to predict match outcomes based on selected features.
    - **Output**: Saves the trained model as a pickle file.
    - **Usage**: Train the model using football match data.

### Predicting Match Outcomes

11. #### `predict_winner.py`
    - **Description**: Loads a trained model and uses it to predict the winner of a match based on provided input data.
    - **Output**: Prints the predicted winner of the match.
    - **Usage**: Make predictions using the loaded model.

## Usage Instructions

1. Clone this repository to your local machine.
2. Install the required Python packages by running `pip install -r requirements.txt`.
3. Ensure you have a compatible web driver (e.g., GeckoDriver for Firefox) installed.
4. Run the scripts in the provided order to execute the entire data analysis pipeline.
5. Customize each script as needed by following the specific usage instructions in their respective sections.

## Contributing

Contributions to this project are highly welcome! Whether it's bug fixes, feature enhancements, or documentation improvements, your input is valuable. Please submit a pull request with your contributions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

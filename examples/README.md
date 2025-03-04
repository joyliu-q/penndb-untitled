# Pipeline #1

Motivation: As the CEO of a company, I want to do competitive analysis on my products to figure out which ones are at risk of being beaten by competitors.

### ETL Process
1. Load data from Joymart (sqlite database with Walmart Kaggle Dataset)
- Generate traditional SQL queries.

2. Load data from FakeAPI Stores
- Store #1 requires hitting a REST API endpoint
- Store #2 requires scraping data from the Walmart website (Not added)

3. Clean the data.
In a real world setting, the names of the products won't be exactly the same. For example, "Granny Smith Apples" in Joymart may be listed as "Red Apples (6oz), Granny Smith" in Walmart.
- For each product in Joymart, find the product in Walmart and Amazon with the closest name.
- For each competitor product, extract the price, reviews (if any), and description.

4. Compare the products.
- For each competitor product, compare the price, reviews (if any), and description to the Joymart product.
- If the competitor product is strictly better, add it to the dataframe.

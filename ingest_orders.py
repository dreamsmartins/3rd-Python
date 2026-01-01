import pandas as pd
import mysql.connector
from datetime import datetime

# Read the CSV file
csv_path = r'c:\Users\UserPC\.conda\3rd Python\Gamezone\gamezone-orders-data.csv'
df = pd.read_csv(csv_path)

# Clean up column names to match our MySQL table
df = df[['USER_ID', 'ORDER_ID', 'PURCHASE_TS', 'SHIP_TS', 'PRODUCT_NAME', 
         'USD_PRICE', 'PURCHASE_PLATFORM', 'MARKETING_CHANNEL', 'ACCOUNT_CREATION_METHOD', 'COUNTRY_CODE']]

df.columns = ['user_id', 'order_id', 'purchase_date', 'ship_date', 'product_name',
              'price', 'purchasing_platform', 'marketing_channel', 'account', 'country_code']

# Convert date strings to datetime objects, handling empty values
df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce').dt.date
df['ship_date'] = pd.to_datetime(df['ship_date'], errors='coerce').dt.date

# Drop rows with invalid dates
df = df.dropna(subset=['purchase_date', 'ship_date'])

# MySQL connection configuration
config = {
    'user': 'root',
    'password': 'fst6Just_Data',  # Replace with your MySQL password
    'host': 'localhost',
    'database': 'gamezone'
}

try:
    # Establish connection to MySQL
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    
    # Insert data row by row
    for idx, row in df.iterrows():
        sql = """INSERT INTO orders 
                (user_id, order_id, purchase_date, ship_date, product_name, 
                 price, purchasing_platform, marketing_channel, account, country_code)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        values = tuple(row)
        cursor.execute(sql, values)
        
        # Commit every 1000 rows
        if idx % 1000 == 0:
            conn.commit()
            print(f"Inserted {idx} rows...")
    
    # Final commit
    conn.commit()
    print("Data ingestion completed successfully!")

except mysql.connector.Error as err:
    print(f"Error: {err}")

finally:
    if 'conn' in locals() and conn.is_connected():
        cursor.close()
        conn.close()
        print("MySQL connection closed.")

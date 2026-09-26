# # Create sqlLite database and load csv into db
# import os
# import sqlite3
# import pandas as pd

# # Paths
# CSV_FILE = r"1_sal_data.csv"
# DB_FOLDER = "database"
# DB_FILE = os.path.join(DB_FOLDER, "employee.db")
# TABLE_NAME = "employees"

# os.makedirs(DB_FOLDER, exist_ok=True)
# df = pd.read_csv(CSV_FILE)
# conn = sqlite3.connect(DB_FILE)
# df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False) # Write DataFrame to SQLite table

# # Verify data
# cursor = conn.cursor()
# cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
# count = cursor.fetchone()[0]
# print(f"Database created successfully.")
# print(f"Table Name : {TABLE_NAME}")
# print(f"Rows Inserted : {count}")







# Query into sqlLite database
import sqlite3
import os
DB_FOLDER = "database"
DB_FILE = os.path.join(DB_FOLDER, "employee.db")
TABLE_NAME = "employees"
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 5")
print("\nSample Records:")
for row in cursor.fetchall():
    print(row)
conn.close()
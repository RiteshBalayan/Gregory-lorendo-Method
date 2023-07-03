import sqlite3
def create_database():
    conn = sqlite3.connect('GL_database.db')  # creates a connection to the database, and if the database does not exist, it is created
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
    CREATE TABLE Prior(
    prior_ID INTEGER PRIMARY KEY,
    r_min REAL,
    r_max REAL,
    w_min REAL,
    w_max REAL,
    w_resolution REAL,
    Bins INTEGER
    )
    """)

    cursor.execute("""
    CREATE TABLE Data(
    Data_ID INTEGER PRIMARY KEY,
    series_ID INTEGER,
    time REAL,
    flux REAL,
    error REAL,
    FOREIGN KEY (series_ID) REFERENCES Data_generator (series_ID)
    )
    """)

    cursor.execute("""
    CREATE TABLE Data_generator(
    series_ID INTEGER PRIMARY KEY,
    frequency REAL,
    phase REAL,
    type_generator TEXT,
    DC REAL,
    noise REAL,
    no_of_data INTEGER,
    time_range REAL,
    sparcity REAL,
    seed INTEGER,
    special_parm_1 REAL
    )
    """)

    cursor.execute("""
    CREATE TABLE GLmethod(
    GL_ID INTEGER PRIMARY KEY,
    prior_ID INTEGER,
    series_ID INTEGER,
    FOREIGN KEY (prior_ID) REFERENCES Prior (prior_ID),
    FOREIGN KEY (series_ID) REFERENCES Data_generator (series_ID)
    )
    """)

    cursor.execute("""
    CREATE TABLE single_integral(
    pw_dm_id INTEGER PRIMARY KEY,
    GL_ID INTEGER,
    m INTEGER,
    w REAL,
    prob REAL,
    FOREIGN KEY (GL_ID) REFERENCES GLmethod (GL_ID)
    )
    """)

    cursor.execute("""
    CREATE TABLE double_integral(
    pd_m_id INTEGER PRIMARY KEY,
    GL_ID INTEGER,
    m INTEGER,
    prob REAL,
    FOREIGN KEY (GL_ID) REFERENCES GLmethod (GL_ID)
    )
    """)

    cursor.execute("""
    CREATE TABLE Marginal_m(
    pw_d_id INTEGER PRIMARY KEY,
    GL_ID INTEGER,
    w REAL,
    prob REAL,
    FOREIGN KEY (GL_ID) REFERENCES GLmethod (GL_ID)
    )
    """)

    conn.commit()  # commits the changes to the database
    conn.close()  # close the connection to the database

create_database()

def print_db_schema():
    conn = sqlite3.connect('GL_database.db')
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in the database:")
    for table_name in tables:
        print(table_name[0])
        cursor.execute(f'PRAGMA table_info({table_name[0]});')
        print("Columns:")
        columns = cursor.fetchall()
        for column in columns:
            print(column)

    conn.close()

print_db_schema()
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import erfc
import matplotlib.pyplot as plt
import time
import sqlite3
from GL_method import GL_calculator

      
        
class SaveGLToDB(GL_calculator):
    '''
    This class save gl_calculated data to sql data base, see database scema in diffrent class.
    Make sure data base is present in same directory

    '''
    def __init__(self, gl_calculator):
        # Call parent constructor
        super().__init__(gl_calculator.data, gl_calculator.priors)
        # Save the state of this instance to the database
        self.save_to_db()

    def save_to_db(self):
        conn = sqlite3.connect('GL_database.db')
        cursor = conn.cursor()

        # Check if the same prior parameters already exist in the Prior table
        cursor.execute("""
            SELECT prior_ID FROM Prior
            WHERE bins = ? AND r_min = ? AND r_max = ? AND w_min = ? AND w_max = ? AND w_resolution = ?
            """, (self.priors['bins'], self.priors['rmin'], self.priors['rmax'],
                  self.priors['w_min'], self.priors['w_max'], self.priors['w_resolution'])
        )
        result = cursor.fetchone()

        if result:
            # Prior parameters already exist, retrieve the prior ID
            self.prior_ID = result[0]
        else:
            # Prior parameters not found, insert a new record
            cursor.execute("""
                INSERT INTO Prior (bins, r_min, r_max, w_min, w_max, w_resolution)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (self.priors['bins'], self.priors['rmin'], self.priors['rmax'],
                      self.priors['w_min'], self.priors['w_max'], self.priors['w_resolution'])
            )
            conn.commit()
            self.prior_ID = cursor.lastrowid

       

        conn.close()

        

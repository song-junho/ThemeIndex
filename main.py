from theme import theme_index
from datetime import datetime

start_date = datetime(2006, 1, 1)
end_date   = datetime.today()

if __name__ == "__main__":

    theme_index.ThemeIndex(start_date, end_date).create_theme_index()
    theme_index.ThemeChgFreq().create_chg_freq()

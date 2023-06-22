from theme import theme_index

start_date = "2006-01-01"
end_date   = "2023-06-30"

if __name__ == "__main__":

    theme_index.ThemeIndex(start_date, end_date).create_theme_index()
    theme_index.ThemeChgFreq().create_chg_freq()

import requests
from bs4 import BeautifulSoup
import pandas as pd

# List of countries with continents and URLs 
countries_to_scrape = [
    {'country': 'Kenya', 'continent': 'Africa', 'url': 'https://www.lonelyplanet.com/kenya/attractions'},
    {'country': 'South Africa', 'continent': 'Africa', 'url': 'https://www.lonelyplanet.com/south-africa/attractions'},
    {'country': 'Egypt', 'continent': 'Africa', 'url': 'https://www.lonelyplanet.com/egypt/attractions'},
    {'country': 'Morocco', 'continent': 'Africa', 'url': 'https://www.lonelyplanet.com/morocco/attractions'},
    {'country': 'Japan', 'continent': 'Asia', 'url': 'https://www.lonelyplanet.com/japan/attractions'},
    {'country': 'China', 'continent': 'Asia', 'url': 'https://www.lonelyplanet.com/china/attractions'},
    {'country': 'India', 'continent': 'Asia', 'url': 'https://www.lonelyplanet.com/india/attractions'},
    {'country': 'Thailand', 'continent': 'Asia', 'url': 'https://www.lonelyplanet.com/thailand/attractions'},
    {'country': 'France', 'continent': 'Europe', 'url': 'https://www.lonelyplanet.com/france/attractions'},
    {'country': 'Italy', 'continent': 'Europe', 'url': 'https://www.lonelyplanet.com/italy/attractions'},
    {'country': 'Germany', 'continent': 'Europe', 'url': 'https://www.lonelyplanet.com/germany/attractions'},
    {'country': 'United Kingdom', 'continent': 'Europe', 'url': 'https://www.lonelyplanet.com/united-kingdom/attractions'},
    {'country': 'United States', 'continent': 'North America', 'url': 'https://www.lonelyplanet.com/usa/attractions'},
    {'country': 'Canada', 'continent': 'North America', 'url': 'https://www.lonelyplanet.com/canada/attractions'},
    {'country': 'Mexico', 'continent': 'North America', 'url': 'https://www.lonelyplanet.com/mexico/attractions'},
    {'country': 'Brazil', 'continent': 'South America', 'url': 'https://www.lonelyplanet.com/brazil/attractions'},
    {'country': 'Argentina', 'continent': 'South America', 'url': 'https://www.lonelyplanet.com/argentina/attractions'},
    {'country': 'Chile', 'continent': 'South America', 'url': 'https://www.lonelyplanet.com/chile/attractions'},
    {'country': 'Peru', 'continent': 'South America', 'url': 'https://www.lonelyplanet.com/peru/attractions'},
    {'country': 'Australia', 'continent': 'Oceania', 'url': 'https://www.lonelyplanet.com/australia/attractions'},
    {'country': 'New Zealand', 'continent': 'Oceania', 'url': 'https://www.lonelyplanet.com/new-zealand/attractions'},
    {'country': 'Fiji', 'continent': 'Oceania', 'url': 'https://www.lonelyplanet.com/fiji/attractions'},
    {'country': 'United Arab Emirates', 'continent': 'Middle East', 'url': 'https://www.lonelyplanet.com/united-arab-emirates/attractions'},
    {'country': 'Turkey', 'continent': 'Middle East', 'url': 'https://www.lonelyplanet.com/turkey/attractions'},
    {'country': 'Israel', 'continent': 'Middle East', 'url': 'https://www.lonelyplanet.com/israel/attractions'},
    {'country': 'Jordan', 'continent': 'Middle East', 'url': 'https://www.lonelyplanet.com/jordan/attractions'},
]

def scrape_country_attractions(country, continent, base_url, num_pages=30):
    data = []
    for page in range(1, num_pages + 1):
        url = f'{base_url}?page={page}'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract attraction titles and descriptions
        titles = soup.find_all('span', class_='heading-05 font-semibold')
        descriptions = soup.find_all('p', class_='relative line-clamp-3')

        # Pair each title with its corresponding description
        for title, description in zip(titles, descriptions):
            data.append({
                'Attraction': title.get_text(strip=True),
                'Description': description.get_text(strip=True),
                'Country': country,
                'Continent': continent
            })

    return data

def main():
    all_data = []
    for entry in countries_to_scrape:
        print(f"Scraping attractions for {entry['country']} in {entry['continent']}...")
        country_data = scrape_country_attractions(entry['country'], entry['continent'], entry['url'])
        all_data.extend(country_data)

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(all_data, columns=['Attraction', 'Description', 'Country', 'Continent'])
    df.to_csv('attractions_data.csv', index=False)
    print("Data saved to attractions_data.csv")

if __name__ == '__main__':
    main()

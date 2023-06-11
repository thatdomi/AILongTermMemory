from bs4 import BeautifulSoup
import requests
from time import sleep
import pandas as pd

class CustomCrawler():
    def __init__(self, name:str, urls:list) -> None:
        self.name = name
        self.urls = urls
    
    def _lazy_init_selenium(self):
        from selenium import webdriver
        self.driver = webdriver.Firefox()
    
    def get_js_webpage(self, url:str) -> str:
        self.driver.get(url)
        sleep(10)
        html = self.driver.page_source
        return html
    
    def get_the_soup(self, html:str):
        soup = BeautifulSoup(html)
        return soup

class AdminCHCrawler(CustomCrawler):
    def __init__(self, name: str, urls: list, export_folder: str, export_default=True) -> None:
        super().__init__(name, urls)
        self.export_folder = export_folder
        self.extracted_law_information = []
        self.export_default = export_default

    def _get_processed_section(self, section):
            processed_section = {}
            section_title = section.find("h1").get_text()
            section_href = section.find("a")["href"]

            processed_section["section_title"] = section_title
            processed_section["section_href"] = section_href

            articles = section.find_all("article")
            article_data = []

            for article in articles:
                article_title = article.find_all("a", fragment=True)[0].get_text()
                article_href = article.find_all("a", fragment=True)[0]["href"]
                article_text = article.get_text().replace("\xa0", "\n")

                article_dict = {
                    "article_title": article_title,
                    "article_href": article_href,
                    "article_text": article_text
                }
                article_data.append(article_dict)

            processed_section["articles"] = article_data
            return processed_section
    
    def get_law_information(self, url, soup):
        # extract gesetzesnummer
        law_information = {}
        law_url = url
        srnummer = soup.find("p", class_="srnummer").get_text()
        erlasstitel = soup.find("h1", class_="erlasstitel botschafttitel").get_text()
        erlasstitel_kurz = soup.find("h2", class_="erlasskurztitel").get_text()

        law_information["url"] = law_url
        law_information["srnummer"] = srnummer
        law_information["erlasstitel"] = erlasstitel
        law_information["erlasstitel_kurz"] = erlasstitel_kurz

        ## process law content
        main = soup.find("main", id="maintext")
        sections = main.find_all("section")
        processed_sections = []
        for section in sections:
            processed_sections.append(self._get_processed_section(section))
        
        law_information["content"] = processed_sections
        self.extracted_law_information.append(law_information)
        
        return law_information

    def convert_internally_stored_law_information_to_df(self) -> pd.DataFrame:
        df_all = pd.DataFrame()
        for extracted_law in self.extracted_law_information:
            df_meta = pd.DataFrame(extracted_law)
            df_sections = pd.json_normalize(df_meta["content"])
            df_full = pd.merge(df_meta, df_sections, left_index=True, right_index=True, how='left')
            df_full_exploded = df_full.explode("articles",ignore_index=True)
            df_articles = pd.json_normalize(df_full_exploded['articles'])
            # Concatenate the expanded columns with the original DataFrame

            df = pd.merge(df_full_exploded, df_articles, left_index=True, right_index=True, how='left')
            df_final = df.drop(columns=['content', 'articles'])

            if self.export_default:
                self._save_law_information_to_disk(df=df_final)

            if len(df_all) == 0:
                df_all = df_final
            else:
                df_all = pd.concat([df_all, df_final])
        return df_all

    def _save_law_information_to_disk(self, df, encoding="UTF-8"):
        filepath = f"{self.export_folder}\\{df['srnummer'][0]}_{df['erlasstitel'][0]}.csv" 
        df.to_csv(filepath, encoding=encoding)
        print(f"file saved to {filepath}")
    
    def full_send(self):
        self._lazy_init_selenium()
        for url in self.urls:
            html = crawler.get_js_webpage(url)
            soup = crawler.get_the_soup(html)
            crawler.get_law_information(url, soup)
            self.convert_internally_stored_law_information_to_df()

        
if __name__ == "__main__":
    EXPORT_PATH = ".\\data\\crawled"
    EXPORT_DEFAULT = True

    name = "admin_ch"
    urls = ['https://www.fedlex.admin.ch/eli/cc/1988/506_506_506/de',
            'https://www.fedlex.admin.ch/eli/cc/1988/517_517_517/de']  # Replace with the URL of the webpage you want to crawl
    #url = 'https://www.fedlex.admin.ch/eli/cc/1988/506_506_506/de'


    crawler = AdminCHCrawler(name=name, urls=urls, export_folder=EXPORT_PATH, export_default=EXPORT_DEFAULT)
    crawler.full_send()
    
    
    
    #crawler._lazy_init_selenium()
    #html = crawler.get_js_webpage(url)
    #soup = crawler.get_the_soup(html)
    #law_information = crawler.get_law_information(soup)


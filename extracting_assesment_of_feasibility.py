""" This file is for getting the project descriptions and 
public interests from xml and save them to a csv file. """
import time
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd


def extract_assesment_of_feasibility(url_df):
    """Function to extract project description and public interest from the xml."""

    #initialize lists
    list_of_lists = []

    #loop through the urls in the csv
    for url in url_df:
        #get project_id from url and save to the list
        project_id = url.split("id=",1)[1]
        list = [project_id]
        

        #get the url content
        url = requests.get(url, timeout=5)
        soup = BeautifulSoup(url.content, 'html.parser')

        #extract the project description from xml and save to the list
        if soup.find("div", {"id": "project-block-reaction"}):
            assesment_of_feasibility = soup.find("div", {"id": "project-block-reaction"})
            assesment_of_feasibility = assesment_of_feasibility.find("tbody")
            assesment_of_feasibility = assesment_of_feasibility.find_all("td")
            count = 0
            for td in assesment_of_feasibility:
                if len(td.text) > 0:
                    list.append(td.text)
                else:
                    print("empty")
                    list.append("EMPTY")

                count += 1
                if count == 3:
                    list_of_lists.append(list)
                    list = [project_id]
                    count = 0

        else:
            list.append("missing assesment of feasibility")
            list.append("missing assesment of feasibility")
            list.append("missing assesment of feasibility")
            list_of_lists.append(list)
        
            print("missing assesment of feasibility")
            
    return list_of_lists
    

def save_to_csv(list_of_lists):
    """Function to save the lists to a csv file."""
    #convert lists to dataframe
    output_df = pd.DataFrame(list_of_lists, columns = ['ID', 'Subjekt', 'Odůvodnění', 'Závěr'])

    #save the dataframe to a csv file
    output_df.to_csv('assesment_of_feasibility.csv', sep=',', encoding='utf-8')


# #convert csv to df
df = pd.read_csv("data\ProjektyPARO_5358953113614861487.csv")

#extract the url list
url_list = df["properties.detail"]

#call the save function
save_to_csv(extract_assesment_of_feasibility(url_list))
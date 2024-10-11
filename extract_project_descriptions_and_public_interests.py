""" This file is for getting the project descritptions and 
public interest from xml and save them to a csv file. """
import time
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd


def extract_project_description_and_public_interest(url_df):
    """Function to extract project description and public interest from the xml."""

    #initialize lists
    list_of_project_ids = []
    list_of_project_descriptions = []
    list_of_public_interest = []

    #loop through the urls in the csv
    for url in url_df:
        #get project_id from url and save to the list
        project_id = url.split("id=",1)[1]
        list_of_project_ids.append(project_id)

        #get the url content
        url = requests.get(url)
        soup = BeautifulSoup(url.content, 'html.parser')

        #extract the project description from xml and save to the list
        project_description = soup.find("div", class_="project-description").text
        project_description = re.sub('\t', '', project_description)
        #project_description = project_description.find("p").text
        list_of_project_descriptions.append(project_description)

        #extract the public interest from xml and save to the list
        project_public_interest = soup.find("div", class_="col-xs-12 col-sm-offset-2 col-sm-8")
        project_public_interest = project_public_interest.find("p")
        project_public_interest = project_public_interest.find("i").text

        project_public_interest = re.sub('\t', '', project_public_interest)
        list_of_public_interest.append(project_public_interest)

        #delay in seconds so we dont get blocked by the site
        DELAY = 3
        time.sleep(DELAY)

    return list_of_project_ids, list_of_project_descriptions, list_of_public_interest

def save_to_csv(list_of_project_ids, list_of_project_descriptions, list_of_public_interest):
    """Function to save the lists to a csv file."""
    #convert lists to dataframe
    output_df = pd.DataFrame(
        {"project_id": list_of_project_ids,
        "project_description": list_of_project_descriptions,
        "public_interest": list_of_public_interest
        })
    print(output_df)
    #save the dataframe to a csv file
    output_df.to_csv('descriptions_and_interests.csv', sep='\t', encoding='utf-8')


#convert csv to df
df = pd.read_csv("data\ProjektyPARO_5358953113614861487.csv")
url_list = df["properties.detail"]
lists = extract_project_description_and_public_interest(url_list)

save_to_csv(lists[0], lists[1], lists[2])

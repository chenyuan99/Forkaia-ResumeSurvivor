import docx2txt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from collections import Counter
#import tkinter as tk
#from tkinter.filedialog import askopenfilename

def process(file):
    # Store the resume in a variable
    #filename = askopenfilename()
    filename = file
    resume = docx2txt.process(filename)

    # Print the resume
    #print(resume)
    stat = dict()

    for filename in os.listdir("./test_job"):
        # Store the job description into a variable
        job_description = docx2txt.process("./test_job/"+filename)

        # Print the job description
        # print(job_description)

        # A list of text
        text = [resume, job_description]

        cv = CountVectorizer()
        count_matrix = cv.fit_transform(text)

        #Print the similarity scores
        #print("\nSimilarity Scores:")
        #print(cosine_similarity(count_matrix))

        #get the match percentage
        matchPercentage = cosine_similarity(count_matrix)[0][1] * 100
        matchPercentage = round(matchPercentage, 2) # round to two decimal
        stat[(resume,filename)] = matchPercentage
        print("Your resume matches about "+ str(matchPercentage)+ "% of the job description:"+ filename)

    match = Counter(stat)
    top3 = match.most_common(3)
    output = 'Your top job recommendations are:'
    for (temp_resume,temp_match) in top3:
        print(temp_resume[1],temp_match,"% matching")
        output += " "+str(temp_resume[1][:-5])+" "+str(temp_match)+" % macthing"
    print(output)
    return output
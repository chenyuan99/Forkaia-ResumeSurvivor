#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 17:04:14 2020

@author: amogh
"""

import io
import os
import re
import nltk
import pandas as pd
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFSyntaxError
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy
from spacy.matcher import Matcher


def get_number_of_pages(file_name):
    try:
        if isinstance(file_name, io.BytesIO):
            # for remote pdf file
            count = 0
            for page in PDFPage.get_pages(
                        file_name,
                        caching=True,
                        check_extractable=True
            ):
                count += 1
            return count
        else:
            # for local pdf file
            if file_name.endswith('.pdf'):
                count = 0
                with open(file_name, 'rb') as fh:
                    for page in PDFPage.get_pages(
                            fh,
                            caching=True,
                            check_extractable=True
                    ):
                        count += 1
                return count
            else:
                return None
    except PDFSyntaxError:
        return None

def extract_text(file_path): 
    text = ''
    for page in extract_text_from_pdf(file_path):
            text += ' ' + page

    return text




def extract_text_from_pdf(pdf_path):
    '''
    Helper function to extract the plain text from .pdf files
    :param pdf_path: path to PDF file to be extracted (remote or local)
    :return: iterator of string of extracted text
    '''
    # https://www.blog.pythonlibrary.org/2018/05/03/exporting-data-from-pdfs-with-python/
    if not isinstance(pdf_path, io.BytesIO):
        # extract text from local pdf file
        with open(pdf_path, 'rb') as fh:
            try:
                for page in PDFPage.get_pages(
                        fh,
                        caching=True,
                        check_extractable=True
                ):
                    resource_manager = PDFResourceManager()
                    fake_file_handle = io.StringIO()
                    converter = TextConverter(
                        resource_manager,
                        fake_file_handle,
                        codec='utf-8',
                        laparams=LAParams()
                    )
                    page_interpreter = PDFPageInterpreter(
                        resource_manager,
                        converter
                    )
                    page_interpreter.process_page(page)

                    text = fake_file_handle.getvalue()
                    yield text

                    # close open handles
                    converter.close()
                    fake_file_handle.close()
            except PDFSyntaxError:
                return
    else:
        # extract text from remote pdf file
        try:
            for page in PDFPage.get_pages(
                    pdf_path,
                    caching=True,
                    check_extractable=True
            ):
                resource_manager = PDFResourceManager()
                fake_file_handle = io.StringIO()
                converter = TextConverter(
                    resource_manager,
                    fake_file_handle,
                    codec='utf-8',
                    laparams=LAParams()
                )
                page_interpreter = PDFPageInterpreter(
                    resource_manager,
                    converter
                )
                page_interpreter.process_page(page)

                text = fake_file_handle.getvalue()
                yield text

                # close open handles
                converter.close()
                fake_file_handle.close()
        except PDFSyntaxError:
            return

"""Pdf to text file"""

#path="/home/amogh/Forkaia/Resume-Survivor/Amogh-Sondur-Resume.pdf"

#path2="/home/amogh/FORKAIA-RESUMES-5.pdf"
#text=extract_text(path)

"""Get number of pages"""

#get_number_of_pages(path)

#text_file = open("resume.txt", "w")
#text_file.write(text)
#text_file.close()



"""Email"""

def extract_email(text):
    '''
    Helper function to extract email id from text
    :param text: plain text extracted from resume file
    '''
    email = re.findall(r"([^@|\s]+@[^@]+\.[^@|\s]+)", text)
    if email:
       return email[0].split()[0].strip(';')
        
            
        
#email=extract_email(text)

"""Number"""


def extract_mobile_number(text):
        mob_num_regex = r'''(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)[-\.\s]*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})'''
        phone = re.findall(re.compile(mob_num_regex), text)
        if phone:
            number = ''.join(phone[0])
        return number


#number=extract_mobile_number(text)


"""Name"""

nlp = spacy.load('en_core_web_sm')

matcher = Matcher(nlp.vocab)


def extract_full_name(nlp_doc):
     pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
     matcher.add('FULL_NAME',None , pattern)
     matches = matcher(nlp_doc)
     for match_id, start, end in matches:
         span = nlp_doc[start:end]
         return span.text

#name=extract_full_name(nlp(text))

"""Links"""


def extract_links(text):
    #^(https?:\/\/)?(www\.)?([a-zA-Z0-9]+(-?[a-zA-Z0-9])*\.)+[\w]{2,}(\/\S*)?$
    links_regx=r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|
    asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an
    |ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|
    cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr
    |ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm
    |jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq
    |mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|
    rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|t
    t|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]
    +\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))"""
    links = re.findall(re.compile(links_regx), text)
    return str(links)

#links=extract_links(text)    



"""Skills"""

def extract_skills(resume_text):
    nlp_text = nlp(resume_text)

    # removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]
    tokens
    
    # reading the csv file
    data = pd.read_csv("/home/amogh/Forkaia/Resume-Survivor/skills.csv") 
    
    # extract values
    skills = list(data.columns.values)
     
    skillset = []
    
    # check for one-grams (example: python)
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)
    doc=nlp(resume_text)
    noun_chunks=doc.noun_chunks        
    # check for bi-grams and tri-grams (example: machine learning)
    for token in doc.noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)
    
    return str(skillset)


#skills=extract_skills(text)


"""Education"""

# Grad all general stop words
STOPWORDS = set(stopwords.words('english'))

# Education Degrees
EDUCATION = [
            'BE','B.E.', 'B.E', 'BS', 'B.S', 
            'ME', 'M.E', 'M.E.', 'MS', 'M.S', 
            'BTECH', 'B.TECH', 'M.TECH', 'MTECH', 
            'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII','Masters','Bachelors'
        ]

def extract_education(resume_text):
    nlp_text = nlp(resume_text)

    # Sentence Tokenizer
    nlp_text = [sent.string.strip() for sent in nlp_text.sents]

    edu = {}
    # Extract education degree
    for index, text in enumerate(nlp_text):
        for tex in text.split():
            # Replace all special symbols
            tex = re.sub(r'[?|$|.|!|]', r'', tex)
            if tex.upper() in EDUCATION and tex not in STOPWORDS:
                edu[tex] = text + nlp_text[index + 1]

    # Extract year
    education = []
    for key in edu.keys():
        year = re.search(re.compile(r'(((20|19)(\d{2})))'), edu[key])
        if year:
            education.append((key, ''.join(year[0])))
        else:
            education.append(key)
    return str(education)


#education=extract_education(text)

"""Experience"""

def extract_company(resume_text):
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # word tokenization
    word_tokens = nltk.word_tokenize(resume_text)

    # remove stop words and lemmatize
    filtered_sentence = [
            w for w in word_tokens if w not
            in stop_words and wordnet_lemmatizer.lemmatize(w)
            not in stop_words
        ]
    sent = nltk.pos_tag(filtered_sentence)

    # parse regex
    cp = nltk.RegexpParser('P: {<NNP>+}')
    cs = cp.parse(sent)

    # for i in cs.subtrees(filter=lambda x: x.label() == 'P'):
    #     print(i)

    test = []

    for vp in list(
        cs.subtrees(filter=lambda x: x.label() == 'P')
    ):
        test.append(" ".join([
            i[0] for i in vp.leaves()
            if len(vp.leaves()) >= 2])
        )

    # Search the word 'experience' in the chunk and
    # then print out the text after it
    x = [
        x[x.lower().index('experience') + 10:]
        for i, x in enumerate(test)
        if x and 'experience' in x.lower()
    ]
    return x


#company=extract_company(text)


"""City"""
def extract_city(text_data):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text_data)
    for ents in doc.ents:
        if(ents.label_ == 'GPE'):
            return (ents.text)	
    else:
        return "null"

"""Entitites"""
RESUME_SECTIONS_GRAD = [
                    'work experience',
                    'experience',
                    'accomplishments',
                    'WORK EXPERIENCE',
                    'experience',
                    'education',
                    'interests',
                    'projects',
                    'professional experience',
                    'publications',
                    'skills',
                    'certifications',
                    'objective',
                    'career objective',
                    'summary',
                    'leadership'
                    
                ]


def extract_entity_sections_grad(text):
    '''
    Helper function to extract all the raw text from sections of
    resume specifically for graduates and undergraduates
    :param text: Raw text of resume
    :return: dictionary of entities
    '''
    text_split = [i.strip() for i in text.split('\n')]
    # sections_in_resume = [i for i in text_split if i.lower() in sections]
    entities = {}
    key = False
    for phrase in text_split:
        if len(phrase) == 1:
            p_key = phrase
        else:
            p_key = set(phrase.lower().split()) & set(RESUME_SECTIONS_GRAD)
        try:
            p_key = list(p_key)[0]
        except IndexError:
            pass
        if p_key in RESUME_SECTIONS_GRAD:
            entities[p_key] = []
            key = p_key
        elif key and phrase.strip():
            entities[key].append(phrase)

    # entity_key = False
    # for entity in entities.keys():
    #     sub_entities = {}
    #     for entry in entities[entity]:
    #         if u'\u2022' not in entry:
    #             sub_entities[entry] = []
    #             entity_key = entry
    #         elif entity_key:
    #             sub_entities[entity_key].append(entry)
    #     entities[entity] = sub_entities

    # pprint.pprint(entities)

    # make entities that are not found None
    # for entity in cs.RESUME_SECTIONS:
    #     if entity not in entities.keys():
    #         entities[entity] = None
    return (entities)



#entites=extract_entity_sections_grad(text)
#entites['projects']

"""
try:
    coll=entites['College Name']
except KeyError:
    pass


try:
    education=entites['education']
except KeyError:
    pass

education
"""
#string_education=(', '.join(education)).lower()
#string_education



"""University"""

"""def extract_university(resume_text):
    nlp_text = nlp(string_education)
    
    
    # removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]
    tokens
    
    # reading the csv file
    univer = pd.read_csv("/home/amogh/universities2.csv") 
    
    # extract values
    uni = list(univer.columns.values)
    stt=str(univer)
    uniset = []
    # check for one-grams (example: python)
    for token in tokens:
        if token.lower() in uni:
            uniset.append(token)
                        
    # check for bi-grams and tri-grams (example: machine learning)
    doc=nlp(string_education)
    doc
    
    noun_chunks=doc.noun_chunks
    noun_chunks    
    for token in doc.noun_chunks:
        token = token.text.lower().strip()
        print(token)
        if token in uni:
            uniset.append(token)
            uniset
    return [i.capitalize() for i in set([i.lower() for i in uniset])]


#university=extract_university(string_education)
#university
"""

#string_education



"""Custom Entities"""

def extract_entities_wih_custom_model(custom_nlp_text):
    '''
    Helper function to extract different entities with custom
    trained model using SpaCy's NER
    :param custom_nlp_text: object of `spacy.tokens.doc.Doc`
    :return: dictionary of entities
    '''
    entities = {}
    for ent in custom_nlp_text.ents:
        if ent.label_ not in entities.keys():
            entities[ent.label_] = [ent.text]
        else:
            entities[ent.label_].append(ent.text)
    for key in entities.keys():
        entities[key] = list(set(entities[key]))
    return entities


#custom_nlp=spacy.load(os.path.dirname(os.path.abspath('/home/amogh/Forkaia/Resume-Survivor/Amogh-Sondur-Resume.pdf')))
#custom_nlp=custom_nlp(text)
#cust_ent=extract_entities_wih_custom_model(custom_nlp)

#cust_ent['work experience']

headers="Name,Email, Phone-no, Links, City, Skills, Education, Degree, Designation, Experience, Projects\n"
out_filename="resume.csv"
f = open(out_filename, "w")
f.write(headers)


def extract_resume(loc):
    directory=loc
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"): 
            path=os.path.join(directory, filename)
                
            file=os.path.splitext(filename)[0]
        
            file_path=directory+'/'+file
            file_path
            
            
            text=extract_text(path)
            custom_nlp=spacy.load(os.path.dirname(os.path.abspath('/home/amogh/Forkaia/Resume-Survivor/Amogh-Sondur-Resume.pdf')))
            custom_nlp=custom_nlp(text)
            cust_ent=extract_entities_wih_custom_model(custom_nlp)
            
            entites=extract_entity_sections_grad(text)

            name="null"
            email="null"
            links="null"
            city="null"
            number="null"
            exp="null"
            education="null"
            designation="null"
            degree="null"
            projects="null"
            
            try:
                name=extract_full_name(nlp(text))
            except (KeyError,IndexError):
                name='null'
                pass           

            try:
                links=extract_links(text)
            except (KeyError,IndexError):
                links='null'
                pass           
            try:
                email=extract_email(text)
            except (KeyError,IndexError):
                email='null'
                pass


            try:
                city=extract_city(text)
            except (KeyError,IndexError,TypeError):
                city='null'
                pass
            try:
                skills=extract_skills(text)
            except (KeyError,IndexError):
                skills='null'
                pass

            try:
                exp=str(entites['experience'])
            except (KeyError,IndexError):
                exp='null'
                pass
            try:
                number=extract_mobile_number(text)
            except (UnboundLocalError,KeyError,IndexError):
                number='null'
                pass
                
            try:
                education=entites['education']
                education=str(education)
            except (KeyError,IndexError):
                education='null'
                pass
            try:
                designation=str(cust_ent['Designation'])
                #designation=str(cust_ent['Role'])
            except (KeyError,IndexError):
                designation='null'
                pass
            try:
                degree=str(cust_ent['Degree'])
            except (KeyError,IndexError):
                degree='null'
                pass
            try:
                projects=str(entites['projects'])
            except (KeyError,IndexError):
                projects='null'
                pass
            
                            
            #print(degree.replace(",","|").replace("\\n","|"))
            #print(projects.replace(",","|").replace("\\n","|"))
            #print(projects)
            #print(name)
            #print(education.replace(",","|").replace("\\n","|"))
            #print(name,email,links, number, city, skills, education, degree, designation, exp, projects)
            #print(designation.replace(",","|"))
            #print(skills.replace(",","|"))
            #headers="Name,Email, Phone-no, Links, City, Skills, Skills, Degree, Designation, Experience, Projects, Companies worked at\n"
            print(file_path)
            #print(city.replace(",","|").replace("\\n","|").replace("\\t","|"))
            f.write(name.replace(",","|").replace("\\n","|").replace("\\t","|")+","+email.replace(",","|").replace("\\n","|").replace("\\t","|")+","+number.replace(",","|").replace("\\n","|").replace("\\t","|")+","+links.replace(",","|").replace("\\n","|").replace("\\t","|") +","+city.replace(",","|").replace("\\n","|").replace("\\t","|")+","+skills.replace(",","|").replace("\\n","|").replace("\\t","|")+","+education.replace(",","|").replace("\\n","|").replace("\\t","|")+","+degree.replace(",","|").replace("\\n","|").replace("\\t","|")+","+designation.replace(",","|").replace("\\n","|").replace("\\t","|")+","+exp.replace(",","|").replace("\\n","|").replace("\\t","|")+","+projects.replace(",","|").replace("\\n","|").replace("\\t","|")+"\n")
            #f.write(projects.replace(",","|").replace("\\n","|").replace("\\t","|")+"\n")
        
"""            record={
                'Name':name,
                'Email':email,
                'Phone-no':number,
                'City':city,
                'Links':links,
                'Education':education,
                'Skills':skills,
                'Degree':degree,
                'Designation':designation,
                'Experience':exp,
                'Projects':projects,
                'Companies Worked At':companies
                }   

            
            import json    
            file+'.json'
            with open(file_path + '.json', 'w') as outfile:
                json.dump(record, outfile)

"""
"""csv file"""

    


extract_resume("/home/amogh/test-res/multi")
f.close()








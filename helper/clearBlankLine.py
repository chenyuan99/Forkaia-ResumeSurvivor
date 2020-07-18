__author__  = "Yuan Chen"
__version__ = "2020.06.16"
import re
import os
from io import open
path0 = "./"
def clearBlankLine():
    file1 = open('text3.txt', 'r', encoding ='utf-8')
    file2 = open('text4.txt', 'w', encoding='utf-8') 
    try:
        for line in file1.readlines():
            if line == '\n':
                line = line.strip("\n")
            file2.write(line)
    finally:
        file1.close()
        file2.close()


if __name__ == '__main__':
    clearBlankLine()

To get running:

First install a virualenv and all the python requirements:
# Setup
`pip install virtualenv` 
`virtualenv venv` 
`source venv/bin/activate` 
`pip install -r requirements.txt` 

Note: Must use python2 - if you get a print error:
  File "Driver.py", line 55
    print "Pickle found at: " + PICKLE_DIR + picklename
virtualenv has installed python3


Required 3rd party downloads:
    -L2 Syntactic Complexity Analyzer from dropbox link (email vadmas@gmail.com for permission). It's been modified to be turned into a python package but user can also download from web here: http://www.personal.psu.edu/xxl13/downloads/l2sca.html
        - Put the L2 syntactic complexity analyzer in the main directory with the following directory structure (and directory names)
            + Driver.py
            + Data
            + SCA
                ├── L2SCA
                │   ├── LICENSE.txt
                │   ├── Makefile
                │   ├── README-L2SCA.txt
                │   ├── README-gui.txt
                │   ├── README-tregex.txt
                │   ├── README-tsurgeon.txt
                │   ├── Semgrex.ppt
                │   ├── **__init__.py**
                │   ├── __init__.pyc
                │   ├── analyzeFolder.py
                │   ├── analyzeText.py
                │   ├── analyzeText.pyc
                │   ├── build.xml
                │   ├── examples
                │   ├── lib
                │   ├── run-tregex-gui.bat
                │   ├── run-tregex-gui.command
                │   ├── samples
                │   ├── stanford-parser-full-2014-01-04
                │   ├── stanford-tregex-3.3.1-javadoc.jar
                │   ├── stanford-tregex-3.3.1-sources.jar
                │   ├── stanford-tregex-3.3.1.jar
                │   ├── stanford-tregex.jar
                │   ├── tregex.sh
                │   └── tsurgeon.sh
                ├── **__init__.py**
            -Note the added __init__.py folders
    -Stanford parser: http://nlp.stanford.edu/software/stanford-parser-full-2015-12-09.zip


-Also download the nltk packages "stopwords", "punkt", "averaged_perceptron_tagger",  using the NLTK Downloader 




# Akshi

## Installation

* Create a folder Akshi and inside the folder
	
	git clone https://github.com/subramgo/Akshi.git

* Download the model files from google drive https://drive.google.com/open?id=1dLg4izlUYVTRkrGyTVv5OJ60awa69zwN
	* from gender folder in google drive, copy the models to  akshi/gender/bin/
	* from face folder in google drive, copy the models to      akshi/face/bin
	* from object folder in google drive, copy the models to akshi/object/bin

* Run setup
	pip3 install -r requirements.txt
	pip install -e .

* Run Akshi
	./run.sh


### Docker

build ourselves
    docker build -t ubuntu-akshi:latest -f Dockerfile.base .
     docker run -p 5000:5000 -d akshi:latest

or from prebuilt image...?
    docker pull subramgo/akshi




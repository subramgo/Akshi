FROM ubuntu-akshi:latest
COPY . /Akshi
WORKDIR /Akshi
RUN pip3 install Pillow==5.1.0
RUN pip3 install requests==2.18.4
RUN pip3 install -e .
ENTRYPOINT ["./run.sh"]

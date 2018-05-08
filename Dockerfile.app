FROM ubuntu-akshi:latest
COPY . /Akshi
WORKDIR /Akshi
RUN pip3 install -e .
ENTRYPOINT ["./run.sh"]

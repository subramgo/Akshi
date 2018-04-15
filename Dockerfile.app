FROM ubuntu-akshi:latest
WORKDIR /Akshi
RUN pip3 install -e .
ENTRYPOINT ["./run.sh"]
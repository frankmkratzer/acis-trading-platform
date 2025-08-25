FROM ubuntu:latest
LABEL authors="frank"

ENTRYPOINT ["top", "-b"]
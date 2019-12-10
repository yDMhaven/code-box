FROM python:3.4-onbuild
RUN apt update || true
RUN apt install -y apt-transport-https
RUN apt install -y node
RUN apt install -y npm
RUN cd ./static && npm install
RUN export FLASK_APP=/usr/src/app/app.py

